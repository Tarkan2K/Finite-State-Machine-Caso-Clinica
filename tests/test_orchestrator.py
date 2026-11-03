"""
Integration tests for ClinicalOrchestrator.
Validates end-to-end clinical note audits, safety intercepts, and FSM terminal failure gating.
"""

from datetime import UTC, datetime

import pytest

from src.exceptions import EngineError
from src.orchestrator import ClinicalOrchestrator, extract_medications
from src.store import ClinicalAuditStore
from src.types import ClinicalNote, ClinicalSafetyStatus, EngineState


@pytest.mark.asyncio
async def test_safe_clinical_note_audit_lifecycle(store: ClinicalAuditStore) -> None:
    """Safe clinical note must complete successfully through all FSM states."""
    orchestrator = ClinicalOrchestrator(store=store)
    note = ClinicalNote(
        note_id="NOTE-SAFE-01",
        patient_ref="PAT-001",
        raw_text="Patient presents with headache. Administer paracetamol 500mg oral.",
        recorded_at=datetime.now(UTC),
    )

    record = await orchestrator.execute_audit(note=note)

    assert record.safety_status == ClinicalSafetyStatus.SAFE
    assert record.fsm_final_state == EngineState.COMPLETED
    assert len(record.alerts) == 0
    assert record.processing_time_ms > 0

    # Ensure record was stored in DuckDB
    persisted = store.get_session(record.session_id)
    assert persisted is not None
    assert persisted["safety_status"] == "SAFE"
    assert persisted["fsm_final_state"] == "COMPLETED"


@pytest.mark.asyncio
async def test_critical_drug_interaction_intercept(store: ClinicalAuditStore) -> None:
    """
    CRITICAL drug-drug interaction must force FSM transition to FAILED state
    and return CRITICAL_BLOCK status to intercept unsafe output.
    """
    orchestrator = ClinicalOrchestrator(store=store)
    note = ClinicalNote(
        note_id="NOTE-CRIT-01",
        patient_ref="PAT-002",
        raw_text="Patient 67M, administer Metformin 500mg BID and schedule contrast CT scan.",
        recorded_at=datetime.now(UTC),
    )

    record = await orchestrator.execute_audit(note=note)

    # Core assertion: FSM must have been forced into FAILED
    assert record.safety_status == ClinicalSafetyStatus.CRITICAL_BLOCK
    assert record.fsm_final_state == EngineState.FAILED
    assert len(record.alerts) == 1
    assert record.alerts[0].severity == "CRITICAL"
    assert "lactic acidosis" in record.alerts[0].description.lower()
    assert "metformin" in record.medications_extracted
    assert "radiocontrast" in record.medications_extracted

    # Verify telemetry in DuckDB
    persisted = store.get_session(record.session_id)
    assert persisted is not None
    assert persisted["safety_status"] == "CRITICAL_BLOCK"
    assert persisted["fsm_final_state"] == "FAILED"


@pytest.mark.asyncio
async def test_audit_execution_timeout_raises(store: ClinicalAuditStore) -> None:
    """Execution exceeding timeout_seconds threshold must raise EngineError."""
    orchestrator = ClinicalOrchestrator(store=store)
    note = ClinicalNote(
        note_id="NOTE-TIMEOUT",
        patient_ref="PAT-003",
        raw_text="Administer aspirin 100mg.",
    )

    # Force immediate timeout with 0.000001 seconds
    with pytest.raises(EngineError) as exc_info:
        await orchestrator.execute_audit(note=note, timeout_seconds=0.000001)
    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_medication_extraction_normalization() -> None:
    """Extractor must normalize entities and parse dosages accurately."""
    text = "Give Metformin 850mg oral and schedule Contrast study"
    meds = await extract_medications(text)

    names = {m.name for m in meds}
    assert "metformin" in names
    assert "radiocontrast" in names  # normalized from contrast

    metformin_med = next(m for m in meds if m.name == "metformin")
    assert metformin_med.dose_mg == 850.0
    assert metformin_med.route == "oral"


@pytest.mark.asyncio
async def test_fentanyl_overdose_intercepted_in_orchestrator(store: ClinicalAuditStore) -> None:
    """Prescription with 200g fentanyl must be intercepted and force FSM to FAILED."""
    orchestrator = ClinicalOrchestrator(store=store)
    note = ClinicalNote(
        note_id="NOTE-FENTANYL-200G",
        patient_ref="12.345.678-5",
        raw_text="fentanilo inyectable a la vena 200 gramos",
        recorded_at=datetime.now(UTC),
    )

    record = await orchestrator.execute_audit(note=note)

    assert record.safety_status == ClinicalSafetyStatus.CRITICAL_BLOCK
    assert record.fsm_final_state == EngineState.FAILED
    assert len(record.alerts) >= 1
    alert = record.alerts[0]
    assert alert.severity == "CRITICAL"
    assert "fentanyl" in record.medications_extracted
