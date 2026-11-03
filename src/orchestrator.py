"""
Asynchronous Clinical Agent Orchestrator powered by local LLM (Qwen 2.5 / Ollama),
durable FSM checkpointing, AST mathematical sandboxing, and DuckDB OLAP telemetry.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timezone, datetime
import json
import logging
import time
from typing import Any
import uuid

import httpx

from src.exceptions import (
    ClinicalSafetyViolationError,
    EngineError,
    ToolSecurityError,
)
from src.fsm import AgentFSM
from src.sandbox import check_dosage_safety, resolve_medication_doses
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

AnalyzerCallable = Callable[
    [dict[str, Any], str],
    Awaitable[tuple[list[ExtractedMedication], list[DrugInteractionAlert]]],
]


async def analyze_clinical_prescription(
    patient: dict[str, Any], raw_text: str
) -> tuple[list[ExtractedMedication], list[DrugInteractionAlert]]:
    """
    Use local LLM pharmacological knowledge to:
    1. Identify all prescribed medications, build their arithmetic daily dose expression in mg
       ('dose_expression'), and specify their maximum safe daily ceiling in mg ('max_safe_daily_mg').
    2. Detect clinical contraindications against the patient's chronic medications, allergies,
       and chronic diseases.
    """
    prompt = (
        "Eres un farmacólogo clínico experto. Analiza la ficha del paciente y la nueva indicación médica.\n"
        f"Ficha del Paciente:\n"
        f"- Edad: {patient.get('edad')} años\n"
        f"- Enfermedades Crónicas: {', '.join(patient.get('enfermedades_cronicas', [])) or 'Ninguna'}\n"
        f"- Alergias Documentadas: {', '.join(patient.get('alergias', [])) or 'Ninguna'}\n"
        f"- Fármacos Crónicos Actuales: {', '.join(patient.get('farmacos_actuales', [])) or 'Ninguno'}\n"
        f"Nueva Indicación Médica: \"{raw_text}\"\n\n"
        "Instrucciones estrictas:\n"
        "1. En 'medications', extrae cada fármaco, sustancia o medio de contraste de la nueva indicación (y los fármacos crónicos relevantes), "
        "normalizando cualquier marca comercial al principio activo genérico en minúsculas.\n"
        "   - 'dose_expression': fórmula aritmética pura en MILIGRAMOS POR DÍA usando solo números y operadores (+, -, *, /, paréntesis). "
        "Si está en gramos multiplica por 1000; si está en mcg divide por 1000; si indica frecuencia (ej. cada 8 horas) multiplica por (24 / horas). "
        "Ejemplo para '20 gramos cada 8 horas': \"20 * 1000 * (24 / 8)\". Ejemplo para '500mg cada 8h': \"500 * (24 / 8)\". Si no hay dosis: null.\n"
        "   - 'max_safe_daily_mg': dosis máxima diaria segura en miligramos para adultos según farmacología clínica "
        "(ej. paracetamol: 4000.0, tramadol: 400.0, metformina: 2550.0, ibuprofeno: 3200.0, aspirina: 4000.0, fentanilo: 0.2, sustancias ilícitas: 0.0).\n"
        "   - 'route': 'oral', 'IV', 'IM', o null.\n"
        "2. En 'alerts', incluye cualquier contraindicación clínica (interacción fármaco-fármaco con sus medicamentos crónicos, "
        "choque con alergias del paciente, o contraindicación por sus enfermedades crónicas como insuficiencia renal o asma). "
        "Cada alerta debe tener 'severity' ('CRITICAL', 'MODERATE' o 'LOW'), 'description' explicando el riesgo fisiopatológico, y 'cie10_codes' (lista de códigos CIE-10).\n\n"
        "Responde ÚNICAMENTE con JSON válido con esta estructura exacta:\n"
        "{\n"
        "  \"medications\": [\n"
        "    {\"name\": \"paracetamol\", \"dose_expression\": \"20 * 1000 * (24 / 8)\", \"max_safe_daily_mg\": 4000.0, \"route\": \"oral\"}\n"
        "  ],\n"
        "  \"alerts\": [\n"
        "    {\"severity\": \"CRITICAL\", \"description\": \"...\", \"cie10_codes\": [\"T36-T50\"]}\n"
        "  ]\n"
        "}"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": ACTIVE_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                raise EngineError(f"Ollama inference error HTTP {resp.status_code}: {resp.text}")

            body = resp.json()
            data = json.loads(body.get("response", "{}").strip())

            meds: list[ExtractedMedication] = []
            for item in data.get("medications", []):
                if isinstance(item, dict) and item.get("name"):
                    expr = item.get("dose_expression")
                    raw_dose = item.get("dose_mg")
                    max_daily = item.get("max_safe_daily_mg")
                    meds.append(
                        ExtractedMedication(
                            name=str(item["name"]).lower().strip(),
                            dose_expression=str(expr).strip() if expr is not None else None,
                            dose_mg=float(raw_dose) if isinstance(raw_dose, (int, float)) else None,
                            max_safe_daily_mg=(
                                float(max_daily) if isinstance(max_daily, (int, float)) else None
                            ),
                            route=item.get("route"),
                        )
                    )

            alerts: list[DrugInteractionAlert] = []
            for al in data.get("alerts", []):
                if isinstance(al, dict) and al.get("description"):
                    sev = str(al.get("severity", "CRITICAL")).upper()
                    if sev not in ("LOW", "MODERATE", "CRITICAL"):
                        sev = "CRITICAL"
                    alerts.append(
                        DrugInteractionAlert(
                            severity=sev,  # type: ignore[arg-type]
                            description=str(al["description"]),
                            cie10_codes=[str(c) for c in al.get("cie10_codes", ["T36-T50"])],
                        )
                    )

            return meds, alerts

    except httpx.RequestError as exc:
        logger.error(f"Ollama LLM connection failed: {exc}")
        raise EngineError(
            f"Clinical AI Ingestion Error: Ollama service at {OLLAMA_API_URL} is unreachable."
        ) from exc
    except json.JSONDecodeError as exc:
        logger.error(f"Malformed LLM JSON output: {exc}")
        raise EngineError(f"Clinical AI Extractor Error: Malformed JSON from LLM: {exc}") from exc


class ClinicalOrchestrator:
    """
    Coordinates end-to-end execution of clinical safety audits under durable FSM governance.
    Persists checkpoints on every state transition and supports resumption from interrupted runs.
    """

    def __init__(
        self,
        store: ClinicalAuditStore,
        analyzer: AnalyzerCallable = analyze_clinical_prescription,
    ) -> None:
        self._store = store
        self._analyzer = analyzer

    async def execute_audit(
        self,
        note: ClinicalNote,
        max_steps: int = 10,
        timeout_seconds: float = 60.0,
        session_id: str | None = None,
    ) -> PatientAuditRecord:
        try:
            return await asyncio.wait_for(
                self._run_pipeline(
                    note=note, max_steps=max_steps, session_id=session_id
                ),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            raise EngineError(
                f"Clinical audit execution timed out after {timeout_seconds}s"
            ) from exc

    async def _run_pipeline(
        self,
        note: ClinicalNote,
        max_steps: int,
        session_id: str | None = None,
    ) -> PatientAuditRecord:
        start_time = time.perf_counter()
        active_session_id = session_id or str(uuid.uuid4())

        # Strict patient lookup in DuckDB EHR registry
        patient = self._store.get_patient(note.patient_ref)
        if not patient:
            raise EngineError(
                f"Paciente con RUT '{note.patient_ref}' no existe en el sistema hospitalario."
            )

        def persist_transition(_record: Any) -> None:
            self._store.save_checkpoint(active_session_id, fsm.to_checkpoint())

        # Rehydrate FSM from checkpoint if resuming an interrupted session
        saved_ckpt = (
            self._store.load_checkpoint(active_session_id) if session_id else None
        )
        if saved_ckpt:
            fsm = AgentFSM.from_checkpoint(saved_ckpt, on_transition=persist_transition)
        else:
            fsm = AgentFSM(max_steps=max_steps, on_transition=persist_transition)
            self._store.save_checkpoint(active_session_id, fsm.to_checkpoint())

        alerts: list[DrugInteractionAlert] = []
        medications: list[ExtractedMedication] = []
        raw_medications: list[ExtractedMedication] = []
        llm_clinical_alerts: list[DrugInteractionAlert] = []
        safety_status = ClinicalSafetyStatus.SAFE

        try:
            # State-driven execution loop supporting durable resumption
            while not fsm.is_terminal:
                state = fsm.current_state

                if state == EngineState.INITIALIZED:
                    fsm.transition(EngineState.PLANNING)

                elif state == EngineState.PLANNING:
                    fsm.transition(EngineState.DISPATCHING_TOOL)

                elif state == EngineState.DISPATCHING_TOOL:
                    ext_start = time.perf_counter()
                    raw_medications, llm_clinical_alerts = await self._analyzer(
                        patient, note.raw_text
                    )
                    ext_latency_ms = (time.perf_counter() - ext_start) * 1000.0
                    fsm.transition(EngineState.AWAITING_OBSERVATION)

                elif state == EngineState.AWAITING_OBSERVATION:
                    ast_start = time.perf_counter()
                    try:
                        medications = resolve_medication_doses(raw_medications)
                    except ToolSecurityError as sec_exc:
                        ast_latency_ms = (time.perf_counter() - ast_start) * 1000.0
                        self._store.write_safety_check(
                            check_id=str(uuid.uuid4()),
                            session_id=active_session_id,
                            tool_name="ast_sandbox_guard",
                            result="CRITICAL",
                            latency_ms=ast_latency_ms,
                            alert_description=str(sec_exc),
                        )
                        alerts.append(
                            DrugInteractionAlert(
                                severity="CRITICAL",
                                description=str(sec_exc),
                                cie10_codes=["T36-T50"],
                            )
                        )
                        safety_status = ClinicalSafetyStatus.CRITICAL_BLOCK
                        fsm.transition(EngineState.FAILED)
                        break

                    for med in medications:
                        self._store.write_medication_span(
                            span_id=str(uuid.uuid4()),
                            session_id=active_session_id,
                            medication=med,
                            extraction_latency_ms=ext_latency_ms if "ext_latency_ms" in locals() else 0.0,
                        )
                    fsm.transition(EngineState.REFLECTING)

                elif state == EngineState.REFLECTING:
                    chk_start = time.perf_counter()
                    dosage_alerts = check_dosage_safety(medications)
                    alerts = llm_clinical_alerts + dosage_alerts
                    chk_latency_ms = (time.perf_counter() - chk_start) * 1000.0

                    has_critical = any(a.severity == "CRITICAL" for a in alerts)
                    self._store.write_safety_check(
                        check_id=str(uuid.uuid4()),
                        session_id=active_session_id,
                        tool_name="clinical_ai_and_ast_sandbox",
                        result="CRITICAL" if has_critical else "SAFE",
                        latency_ms=chk_latency_ms,
                        alert_description=alerts[0].description if alerts else None,
                    )

                    if has_critical:
                        safety_status = ClinicalSafetyStatus.CRITICAL_BLOCK
                        fsm.transition(EngineState.FAILED)
                    else:
                        fsm.transition(EngineState.COMPLETED)

        except ClinicalSafetyViolationError:
            safety_status = ClinicalSafetyStatus.CRITICAL_BLOCK
            if not fsm.is_terminal:
                fsm.transition(EngineState.FAILED)
        except Exception:
            safety_status = ClinicalSafetyStatus.CRITICAL_BLOCK
            if not fsm.is_terminal:
                fsm.transition(EngineState.FAILED)
            raise

        total_processing_ms = (time.perf_counter() - start_time) * 1000.0

        audit_record = PatientAuditRecord(
            session_id=active_session_id,
            patient_ref=note.patient_ref,
            note_text=note.raw_text,
            medications_extracted=[m.name for m in medications],
            safety_status=safety_status,
            alerts=alerts,
            fsm_final_state=fsm.current_state,
            processing_time_ms=total_processing_ms,
            created_at=datetime.now(timezone.utc),
        )

        self._store.write_session(audit_record)
        return audit_record
