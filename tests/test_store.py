"""
Unit and integration tests for the DuckDB analytical store (ClinicalAuditStore).
Validates relational persistence, schema constraints, and OLAP latency percentiles.
"""

from datetime import datetime, timezone
import uuid
import pytest
from src.exceptions import StorageError
from src.store import ClinicalAuditStore
from src.types import (
    ClinicalSafetyStatus,
    EngineState,
    ExtractedMedication,
    PatientAuditRecord,
)


def _create_dummy_record(
    session_id: str | None = None,
    status: ClinicalSafetyStatus = ClinicalSafetyStatus.SAFE,
) -> PatientAuditRecord:
    """Helper factory for generating valid audit records."""
    return PatientAuditRecord(
        session_id=session_id or str(uuid.uuid4()),
        patient_ref="PAT-TEST-001",
        note_text="Administer paracetamol 500mg oral",
        medications_extracted=["paracetamol"],
        safety_status=status,
        alerts=[],
        fsm_final_state=EngineState.COMPLETED,
        processing_time_ms=12.4,
        created_at=datetime.now(timezone.utc),
    )


def test_session_write_and_read(store: ClinicalAuditStore) -> None:
    """Audit record must be successfully persisted and retrieved by session_id."""
    record = _create_dummy_record()
    store.write_session(record)

    persisted = store.get_session(record.session_id)
    assert persisted is not None
    assert persisted["session_id"] == record.session_id
    assert persisted["patient_ref"] == record.patient_ref
    assert persisted["safety_status"] == record.safety_status.value
    assert persisted["fsm_final_state"] == record.fsm_final_state.value


def test_nonexistent_session_returns_none(store: ClinicalAuditStore) -> None:
    """Querying an unknown session_id must return None."""
    assert store.get_session("nonexistent-uuid") is None


def test_duplicate_session_raises_storage_error(store: ClinicalAuditStore) -> None:
    """Primary key violation must raise StorageError."""
    record = _create_dummy_record()
    store.write_session(record)

    with pytest.raises(StorageError) as exc_info:
        # Re-inserting the same session_id must trigger PK collision
        store.write_session(record)
    assert "Failed to persist clinical session" in str(exc_info.value)


def test_medication_span_persistence(store: ClinicalAuditStore) -> None:
    """Medication spans must be recorded with latency telemetry."""
    session = _create_dummy_record()
    store.write_session(session)

    med = ExtractedMedication(name="metformin", dose_mg=500.0, route="oral")
    store.write_medication_span(
        span_id=str(uuid.uuid4()),
        session_id=session.session_id,
        medication=med,
        extraction_latency_ms=3.25,
    )


def test_olap_latency_percentiles_calculation(store: ClinicalAuditStore) -> None:
    """v_safety_performance view must calculate P50, P95, and P99 percentiles."""
    session = _create_dummy_record()
    store.write_session(session)

    # Insert a sequence of latency observations: 1.0, 2.0, 3.0, 4.0, 10.0 ms
    latencies = [1.0, 2.0, 3.0, 4.0, 10.0]
    for lat in latencies:
        store.write_safety_check(
            check_id=str(uuid.uuid4()),
            session_id=session.session_id,
            tool_name="interaction_checker",
            result="SAFE",
            latency_ms=lat,
        )

    metrics = store.get_safety_performance()
    assert len(metrics) == 1
    row = metrics[0]
    assert row["tool_name"] == "interaction_checker"
    assert row["total_checks"] == 5
    assert row["p50_ms"] == pytest.approx(3.0, rel=0.1)
    assert row["p99_ms"] >= 4.0