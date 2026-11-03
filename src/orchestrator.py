"""
Asynchronous Clinical Agent Orchestrator.
Coordinates FSM lifecycle, tool execution, safety sandboxing, dosage ceiling checks, and DuckDB telemetry.
"""

import asyncio
import os
import re
import time
import uuid
from datetime import UTC, datetime

import httpx

from src.exceptions import ClinicalSafetyViolationError, EngineError
from src.fsm import AgentFSM
from src.sandbox import check_dosage_safety, check_drug_interactions
from src.store import ClinicalAuditStore
from src.types import (
    ClinicalNote,
    ClinicalSafetyStatus,
    DrugInteractionAlert,
    EngineState,
    ExtractedMedication,
    PatientAuditRecord,
)

# Known active substances regex (bilingual: English + Spanish clinical variants)
KNOWN_DRUGS_PATTERN = re.compile(
    r"\b(metformin|metformina|radiocontrast|contrast|contraste|warfarin|warfarina|aspirin|aspirina|tramadol|ssri|methotrexate|nsaid|paracetamol|ibuprofen|ibuprofeno|cocaine|cocaina|cocaína|fentanyl|fentanilo)\b",
    re.IGNORECASE,
)

# Flexible dosage regex matching grams, milligrams, micrograms in English and Spanish
DOSE_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|miligramos|miligramo|mcg|microgramos|g|gr|gramos|gramo)\b",
    re.IGNORECASE,
)

# Bilingual normalization dictionary to active pharmaceutical ingredients
NORMALIZATION_MAP = {
    "metformina": "metformin",
    "contrast": "radiocontrast",
    "contraste": "radiocontrast",
    "warfarina": "warfarin",
    "aspirina": "aspirin",
    "ibuprofeno": "ibuprofen",
    "cocaina": "cocaine",
    "cocaína": "cocaine",
    "fentanilo": "fentanyl",
}


def _parse_dose_to_mg(text: str) -> float | None:
    """Parse dosage strings and normalize units to milligrams."""
    match = DOSE_PATTERN.search(text)
    if not match:
        return None

    raw_val_str = match.group(1).replace(",", ".")
    unit = match.group(2).lower().strip()

    try:
        val = float(raw_val_str)
    except ValueError:
        return None

    if unit in ("g", "gr", "gramos", "gramo"):
        return val * 1000.0  # Convert grams to mg
    if unit in ("mcg", "microgramos"):
        return val / 1000.0  # Convert mcg to mg
    return val  # Default is mg


async def extract_medications(raw_text: str) -> list[ExtractedMedication]:
    """
    Extract medication entities and normalize dosage to milligrams.
    If OLLAMA_HOST is set, queries local Ollama; otherwise uses fast bilingual extractor.
    """
    await asyncio.sleep(0.005)

    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        try:
            prompt = (
                f"Extract all medications as JSON list: [{{\"name\": \"drug\", \"dose_mg\": 500}}]. Text: \"{raw_text}\""
            )
            async with httpx.AsyncClient(timeout=1.0) as client:
                res = await client.post(
                    f"{ollama_host}/api/generate",
                    json={"model": "llama3.2", "prompt": prompt, "stream": False, "format": "json"},
                )
                if res.status_code == 200:
                    import json
                    parsed = json.loads(res.json().get("response", "[]"))
                    if isinstance(parsed, list) and parsed:
                        return [
                            ExtractedMedication(
                                name=NORMALIZATION_MAP.get(str(p["name"]).lower().strip(), str(p["name"]).lower().strip()),
                                dose_mg=float(p.get("dose_mg")) if p.get("dose_mg") else None,
                            )
                            for p in parsed if "name" in p
                        ]
        except (httpx.RequestError, KeyError, ValueError):
            pass

    # Deterministic bilingual extraction with unit normalization
    matches = KNOWN_DRUGS_PATTERN.findall(raw_text)
    calculated_dose_mg = _parse_dose_to_mg(raw_text)

    extracted: list[ExtractedMedication] = []
    seen = set()

    for match in matches:
        raw_name = match.lower().strip()
        canonical_name = NORMALIZATION_MAP.get(raw_name, raw_name)

        if canonical_name not in seen:
            seen.add(canonical_name)
            extracted.append(
                ExtractedMedication(
                    name=canonical_name,
                    dose_mg=calculated_dose_mg,
                    route="oral" if ("oral" in raw_text.lower() or "boca" in raw_text.lower()) else "injected" if "inyectado" in raw_text.lower() else None,
                )
            )

    return extracted


class ClinicalOrchestrator:
    """
    Coordinates end-to-end execution of clinical safety audits under FSM governance.
    """

    def __init__(self, store: ClinicalAuditStore) -> None:
        self._store = store

    async def execute_audit(
        self,
        note: ClinicalNote,
        max_steps: int = 10,
        timeout_seconds: float = 30.0,
    ) -> PatientAuditRecord:
        """
        Execute an end-to-end clinical safety audit for a submitted clinical note.
        """
        try:
            return await asyncio.wait_for(
                self._run_pipeline(note=note, max_steps=max_steps),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            raise EngineError(
                f"Clinical audit execution timed out after {timeout_seconds}s"
            ) from exc

    async def _run_pipeline(
        self, note: ClinicalNote, max_steps: int
    ) -> PatientAuditRecord:
        start_time = time.perf_counter()
        session_id = str(uuid.uuid4())
        fsm = AgentFSM(max_steps=max_steps)

        alerts: list[DrugInteractionAlert] = []
        medications: list[ExtractedMedication] = []
        safety_status = ClinicalSafetyStatus.SAFE

        try:
            # Step 1: INITIALIZED -> PLANNING
            fsm.transition(EngineState.PLANNING)

            # Step 2: PLANNING -> DISPATCHING_TOOL (Medication Extractor)
            fsm.transition(EngineState.DISPATCHING_TOOL)
            ext_start = time.perf_counter()
            medications = await extract_medications(note.raw_text)
            ext_latency_ms = (time.perf_counter() - ext_start) * 1000.0

            # Step 3: DISPATCHING_TOOL -> AWAITING_OBSERVATION
            fsm.transition(EngineState.AWAITING_OBSERVATION)

            # Record extracted spans in DuckDB
            for med in medications:
                span_id = str(uuid.uuid4())
                self._store.write_medication_span(
                    span_id=span_id,
                    session_id=session_id,
                    medication=med,
                    extraction_latency_ms=ext_latency_ms,
                )

            # Step 4: AWAITING_OBSERVATION -> REFLECTING (Multi-vector Safety Validation)
            fsm.transition(EngineState.REFLECTING)
            med_names = [m.name for m in medications]

            chk_start = time.perf_counter()
            # 1. Check drug-drug contraindications
            interaction_alerts = check_drug_interactions(med_names)
            # 2. Check lethal dosage and posology ceilings (Overdose Guardrail)
            dosage_alerts = check_dosage_safety(medications)
            alerts = interaction_alerts + dosage_alerts

            chk_latency_ms = (time.perf_counter() - chk_start) * 1000.0

            # Log check result telemetry
            check_id = str(uuid.uuid4())
            has_critical = any(a.severity == "CRITICAL" for a in alerts)

            self._store.write_safety_check(
                check_id=check_id,
                session_id=session_id,
                tool_name="clinical_safety_matrix",
                result="CRITICAL" if has_critical else "SAFE",
                latency_ms=chk_latency_ms,
                alert_description=alerts[0].description if alerts else None,
            )

            # Step 5: Evaluate safety alerts and transition FSM
            if has_critical:
                safety_status = ClinicalSafetyStatus.CRITICAL_BLOCK
                # Intercept execution: force transition to terminal FAILED state
                fsm.transition(EngineState.FAILED)
            else:
                fsm.transition(EngineState.COMPLETED)

        except ClinicalSafetyViolationError:
            safety_status = ClinicalSafetyStatus.CRITICAL_BLOCK
            if fsm.current_state != EngineState.FAILED:
                fsm.transition(EngineState.FAILED)
        except Exception:
            safety_status = ClinicalSafetyStatus.CRITICAL_BLOCK
            if fsm.current_state not in (EngineState.COMPLETED, EngineState.FAILED):
                fsm.transition(EngineState.FAILED)
            raise

        total_processing_ms = (time.perf_counter() - start_time) * 1000.0

        # Construct immutable audit record
        audit_record = PatientAuditRecord(
            session_id=session_id,
            patient_ref=note.patient_ref,
            note_text=note.raw_text,
            medications_extracted=[m.name for m in medications],
            safety_status=safety_status,
            alerts=alerts,
            fsm_final_state=fsm.current_state,
            processing_time_ms=total_processing_ms,
            created_at=datetime.now(UTC),
        )

        # Persist session audit trail to DuckDB
        self._store.write_session(audit_record)

        return audit_record
