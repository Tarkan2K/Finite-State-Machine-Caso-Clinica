"""
Integration tests for ClinicalOrchestrator.
Validates end-to-end clinical note audits, AST security interception, FSM checkpoint resumption, and failure gating.
"""

from datetime import timezone, datetime

import pytest

from src.exceptions import EngineError
from src.orchestrator import ClinicalOrchestrator
from src.store import ClinicalAuditStore
from src.types import ClinicalNote, ClinicalSafetyStatus, EngineState
from tests.conftest import deterministic_test_analyzer


@pytest.mark.asyncio
async def test_unregistered_rut_raises_engine_error(store: ClinicalAuditStore) -> None:
    """Auditing an unregistered RUT must raise EngineError."""
    orchestrator = ClinicalOrchestrator(store=store, analyzer=deterministic_test_analyzer)
    note = ClinicalNote(
        note_id="NOTE-INVALID-RUT",
        patient_ref="99.999.999-9",
        raw_text="Administrar paracetamol 500mg oral.",
        recorded_at=datetime.now(timezone.utc),
    )

    with pytest.raises(EngineError) as exc_info:
        await orchestrator.execute_audit(note=note)
    assert "no existe en el sistema hospitalario" in str(exc_info.value)


@pytest.mark.asyncio
async def test_safe_clinical_note_audit_lifecycle(store: ClinicalAuditStore) -> None:
    """Safe clinical note must complete successfully through all FSM states and save checkpoints."""
    orchestrator = ClinicalOrchestrator(store=store, analyzer=deterministic_test_analyzer)
    note = ClinicalNote(
        note_id="NOTE-SAFE-01",
        patient_ref="18.901.234-K",
        raw_text="Administer paracetamol 500mg oral.",
        recorded_at=datetime.now(timezone.utc),
    )

    record = await orchestrator.execute_audit(note=note)

    assert record.safety_status == ClinicalSafetyStatus.SAFE
    assert record.fsm_final_state == EngineState.COMPLETED
    assert len(record.alerts) == 0

    ckpt = store.load_checkpoint(record.session_id)
    assert ckpt is not None
    assert ckpt["current_state"] == "COMPLETED"


@pytest.mark.asyncio
async def test_paracetamol_20g_overdose_intercept(store: ClinicalAuditStore) -> None:
    """20 grams of paracetamol every 8 hours (60,000 mg/day) must force FSM to FAILED."""
    orchestrator = ClinicalOrchestrator(store=store, analyzer=deterministic_test_analyzer)
    note = ClinicalNote(
        note_id="NOTE-OVERDOSE-20G",
        patient_ref="12.345.678-5",
        raw_text="Administrar 20 gramos de paracetamol cada 8 horas",
        recorded_at=datetime.now(timezone.utc),
    )

    record = await orchestrator.execute_audit(note=note)

    assert record.safety_status == ClinicalSafetyStatus.CRITICAL_BLOCK
    assert record.fsm_final_state == EngineState.FAILED
    assert len(record.alerts) == 1
    assert "60000.0 mg/día" in record.alerts[0].description


@pytest.mark.asyncio
async def test_ast_sandbox_security_attack_intercepted_and_logged(
    store: ClinicalAuditStore,
) -> None:
    """Malicious code in dose_expression must be caught by AST sandbox, logged in DuckDB, and force FSM to FAILED."""
    orchestrator = ClinicalOrchestrator(store=store, analyzer=deterministic_test_analyzer)
    note = ClinicalNote(
        note_id="NOTE-AST-ATTACK",
        patient_ref="18.901.234-K",
        raw_text="Inject __import__('os').system('echo pwned')",
        recorded_at=datetime.now(timezone.utc),
    )

    record = await orchestrator.execute_audit(note=note)

    assert record.safety_status == ClinicalSafetyStatus.CRITICAL_BLOCK
    assert record.fsm_final_state == EngineState.FAILED
    assert any("SECURITY BLOCKED" in a.description for a in record.alerts)

    metrics = store.get_safety_performance()
    assert any(m["tool_name"] == "ast_sandbox_guard" for m in metrics)
