"""
Asynchronous Clinical Agent Orchestrator powered by local LLM (Qwen 2.5 / Ollama).
Coordinates LLM semantic extraction, FSM lifecycle, AST sandboxing, dosage ceilings, and DuckDB telemetry.
"""

import asyncio
import json
import logging
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

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
ACTIVE_LLM_MODEL = "qwen2.5:3b"

# Active patient catalog preserving true clinical baselines
PATIENT_REGISTRY = {
    "PAT-101": {
        "name": "Eleanor Vance",
        "chronic_medications": [
            ExtractedMedication(name="metformin", dose_mg=850.0, route="oral")
        ],
    },
    "PAT-102": {
        "name": "Carlos Silva",
        "chronic_medications": [],
    },
    "PAT-103": {
        "name": "Arthur Pendelton",
        "chronic_medications": [
            ExtractedMedication(name="warfarin", dose_mg=5.0, route="oral")
        ],
    },
}

KNOWN_SUBSTANCES = [
    "metformin", "metformina", "radiocontrast", "contrast", "contraste",
    "warfarin", "warfarina", "aspirin", "aspirina", "tramadol", "ssri",
    "methotrexate", "nsaid", "paracetamol", "ibuprofen", "ibuprofeno",
    "cocaine", "cocaina", "cocaína", "fentanyl", "fentanilo",
]

NORMALIZATION_MAP = {
    "metformina": "metformin",
    "contrast": "radiocontrast",
    "contraste": "radiocontrast",
    "medio de contraste": "radiocontrast",
    "warfarina": "warfarin",
    "aspirina": "aspirin",
    "ibuprofeno": "ibuprofen",
    "paracetamol": "paracetamol",
    "acetaminofen": "paracetamol",
    "acetaminofén": "paracetamol",
    "tylenol": "paracetamol",
    "tramadol": "tramadol",
    "cocaina": "cocaine",
    "cocaína": "cocaine",
    "fentanilo": "fentanyl",
    "metotrexato": "methotrexate",
    "aine": "nsaid",
    "aines": "nsaid",
    "isrs": "ssri",
}


def _span_based_fallback_extraction(raw_text: str) -> list[ExtractedMedication]:
    """
    Syntactic span-based fallback for CI environments where Ollama is offline.
    Associates dosage amounts strictly to the adjacent medication name.
    """
    extracted: list[ExtractedMedication] = []
    seen = set()

    # Pattern finding drug and its immediately adjacent dosage (before or after)
    for drug in KNOWN_SUBSTANCES:
        # Match e.g. "10 gramos de paracetamol" or "paracetamol 500mg"
        pattern = re.compile(
            rf"(?:(\d+(?:[.,]\d+)?)\s*(mg|miligramos|g|gr|gramos|mcg)\s+(?:de\s+)?)?\b{drug}\b(?:\s+(?:de\s+)?(\d+(?:[.,]\d+)?)\s*(mg|miligramos|g|gr|gramos|mcg))?",
            re.IGNORECASE,
        )
        match = pattern.search(raw_text)
        if match:
            canonical = NORMALIZATION_MAP.get(drug.lower(), drug.lower())
            if canonical in seen:
                continue
            seen.add(canonical)

            raw_val = match.group(1) or match.group(3)
            unit = (match.group(2) or match.group(4) or "mg").lower()
            dose_mg = None

            if raw_val:
                try:
                    val = float(raw_val.replace(",", "."))
                    if unit in ("g", "gr", "gramos"):
                        dose_mg = val * 1000.0
                    elif unit in ("mcg",):
                        dose_mg = val / 1000.0
                    else:
                        dose_mg = val
                except ValueError:
                    dose_mg = None

            route = "injected" if "inyectado" in raw_text.lower() else "oral" if "oral" in raw_text.lower() else None
            extracted.append(ExtractedMedication(name=canonical, dose_mg=dose_mg, route=route))

    return extracted


async def extract_medications(raw_text: str) -> list[ExtractedMedication]:
    """
    Extract active medications, substances, and convert dosages using real LLM reasoning.
    Queries local Qwen 2.5 via Ollama; falls back to span-based parser if Ollama is unreachable.
    """
    prompt = (
        "You are an expert clinical pharmacologist entity extractor. "
        "Analyze this physician bedside note and extract every medication, contrast agent, or substance mentioned. "
        "For each substance, identify: "
        "- name: normalized canonical active substance in lowercase (e.g., 'paracetamol', 'metformin', 'tramadol', 'radiocontrast', 'aspirin', 'cocaine', 'fentanyl'). "
        "- dose_mg: exact quantity converted to MILLIGRAMS (e.g., 5g = 5000, 500mg = 500, 10 gramos = 10000). If no dose mentioned, return null. "
        "- route: 'oral', 'injected', 'IV', or null. "
        f"Physician Note: \"{raw_text}\"\n"
        "Return ONLY a JSON list of objects. Example: [{\"name\": \"paracetamol\", \"dose_mg\": 10000, \"route\": \"oral\"}]\n"
        "JSON:"
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": ACTIVE_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            if resp.status_code == 200:
                body = resp.json()
                raw_json = body.get("response", "[]").strip()
                data = json.loads(raw_json)

                if isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            data = v
                            break

                if isinstance(data, list) and data:
                    meds: list[ExtractedMedication] = []
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            raw_name = str(item["name"]).lower().strip()
                            med_name = NORMALIZATION_MAP.get(raw_name, raw_name)
                            dose_val = item.get("dose_mg")
                            try:
                                dose_flt = float(dose_val) if dose_val is not None else None
                            except (ValueError, TypeError):
                                dose_flt = None

                            meds.append(
                                ExtractedMedication(
                                    name=med_name,
                                    dose_mg=dose_flt,
                                    route=item.get("route"),
                                )
                            )
                    if meds:
                        return meds
    except (httpx.RequestError, json.JSONDecodeError, KeyError) as exc:
        logger.debug(f"Ollama request skipped or failed: {exc}")

    # Accurate span-based fallback for offline CI execution
    return _span_based_fallback_extraction(raw_text)


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
        safety_status = ClinicalSafetyStatus.SAFE

        try:
            # Step 1: INITIALIZED -> PLANNING
            fsm.transition(EngineState.PLANNING)

            # Step 2: PLANNING -> DISPATCHING_TOOL (True LLM Extractor)
            fsm.transition(EngineState.DISPATCHING_TOOL)
            ext_start = time.perf_counter()
            new_medications = await extract_medications(note.raw_text)
            ext_latency_ms = (time.perf_counter() - ext_start) * 1000.0

            # Reconcile with patient's real EHR chronic medications
            patient_profile = PATIENT_REGISTRY.get(note.patient_ref, {"chronic_medications": []})
            chronic_meds = patient_profile["chronic_medications"]

            # Combined medication pool (chronic baseline + new order)
            all_medications = list(chronic_meds) + list(new_medications)

            # Step 3: DISPATCHING_TOOL -> AWAITING_OBSERVATION
            fsm.transition(EngineState.AWAITING_OBSERVATION)

            # Record extracted spans in DuckDB
            for med in all_medications:
                span_id = str(uuid.uuid4())
                self._store.write_medication_span(
                    span_id=span_id,
                    session_id=session_id,
                    medication=med,
                    extraction_latency_ms=ext_latency_ms,
                )

            # Step 4: AWAITING_OBSERVATION -> REFLECTING (Multi-vector Safety Validation)
            fsm.transition(EngineState.REFLECTING)
            med_names = [m.name for m in all_medications]

            chk_start = time.perf_counter()
            # 1. Check drug-drug contraindications
            interaction_alerts = check_drug_interactions(med_names)
            # 2. Check lethal dosage ceilings (Overdose Guardrail)
            dosage_alerts = check_dosage_safety(all_medications)
            alerts = interaction_alerts + dosage_alerts

            chk_latency_ms = (time.perf_counter() - chk_start) * 1000.0

            # Log check result telemetry
            check_id = str(uuid.uuid4())
            has_critical = any(a.severity == "CRITICAL" for a in alerts)

            self._store.write_safety_check(
                check_id=check_id,
                session_id=session_id,
                tool_name="qwen_clinical_safety_matrix",
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
            medications_extracted=[m.name for m in all_medications],
            safety_status=safety_status,
            alerts=alerts,
            fsm_final_state=fsm.current_state,
            processing_time_ms=total_processing_ms,
            created_at=datetime.now(UTC),
        )

        # Persist session audit trail to DuckDB
        self._store.write_session(audit_record)

        return audit_record
