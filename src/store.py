"""
DuckDB analytical storage layer for clinical sessions, telemetry, and audit trails.
Provides high-performance columnar ingestion and real-time latency percentile views.
"""

from datetime import UTC, datetime
from typing import Any

import duckdb

from src.exceptions import StorageError
from src.types import ExtractedMedication, PatientAuditRecord


class ClinicalAuditStore:
    """
    Embedded analytical audit store powered by DuckDB.
    Persists clinical sessions, extracted entities, and safety check telemetry.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """
        Initialize the DuckDB connection and create database schema.

        Args:
            db_path: Path to database file or ':memory:' for transient storage.
        """
        try:
            self._conn = duckdb.connect(db_path)
            self._initialize_schema()
        except Exception as exc:
            raise StorageError(f"Failed to initialize DuckDB storage: {exc}") from exc

    def _initialize_schema(self) -> None:
        """Create relational tables and analytical views."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS clinical_sessions (
                session_id VARCHAR PRIMARY KEY,
                patient_ref VARCHAR NOT NULL,
                note_text TEXT NOT NULL,
                safety_status VARCHAR NOT NULL,
                fsm_final_state VARCHAR NOT NULL,
                processing_time_ms DOUBLE NOT NULL,
                created_at VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS medication_spans (
                span_id VARCHAR PRIMARY KEY,
                session_id VARCHAR NOT NULL,
                medication_name VARCHAR NOT NULL,
                dose_mg DOUBLE,
                route VARCHAR,
                extraction_latency_ms DOUBLE NOT NULL,
                created_at VARCHAR NOT NULL
            );

            CREATE TABLE IF NOT EXISTS safety_checks (
                check_id VARCHAR PRIMARY KEY,
                session_id VARCHAR NOT NULL,
                tool_name VARCHAR NOT NULL,
                result VARCHAR NOT NULL,
                latency_ms DOUBLE NOT NULL,
                alert_description TEXT,
                checked_at VARCHAR NOT NULL
            );

            CREATE OR REPLACE VIEW v_safety_performance AS
            SELECT
                tool_name,
                COUNT(*) AS total_checks,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_ms
            FROM safety_checks
            GROUP BY tool_name;
            """
        )

    def write_session(self, record: PatientAuditRecord) -> None:
        """
        Persist a finalized patient audit record to the store.

        Args:
            record: Immutable PatientAuditRecord to write.
        """
        try:
            self._conn.execute(
                """
                INSERT INTO clinical_sessions (
                    session_id, patient_ref, note_text, safety_status,
                    fsm_final_state, processing_time_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    record.session_id,
                    record.patient_ref,
                    record.note_text,
                    record.safety_status.value,
                    record.fsm_final_state.value,
                    record.processing_time_ms,
                    record.created_at.isoformat(),
                ],
            )
        except Exception as exc:
            raise StorageError(f"Failed to persist clinical session: {exc}") from exc

    def write_medication_span(
        self,
        span_id: str,
        session_id: str,
        medication: ExtractedMedication,
        extraction_latency_ms: float,
    ) -> None:
        """Record an extracted medication entity with its latency trace."""
        try:
            self._conn.execute(
                """
                INSERT INTO medication_spans (
                    span_id, session_id, medication_name, dose_mg, route,
                    extraction_latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    span_id,
                    session_id,
                    medication.name,
                    medication.dose_mg,
                    medication.route,
                    extraction_latency_ms,
                    datetime.now(UTC).isoformat(),
                ],
            )
        except Exception as exc:
            raise StorageError(f"Failed to record medication span: {exc}") from exc

    def write_safety_check(
        self,
        check_id: str,
        session_id: str,
        tool_name: str,
        result: str,
        latency_ms: float,
        alert_description: str | None = None,
    ) -> None:
        """Record an individual safety validation check execution and its latency."""
        try:
            self._conn.execute(
                """
                INSERT INTO safety_checks (
                    check_id, session_id, tool_name, result, latency_ms,
                    alert_description, checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    check_id,
                    session_id,
                    tool_name,
                    result,
                    latency_ms,
                    alert_description,
                    datetime.now(UTC).isoformat(),
                ],
            )
        except Exception as exc:
            raise StorageError(f"Failed to record safety check: {exc}") from exc

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a clinical session record by unique session ID."""
        cursor = self._conn.execute(
            """
            SELECT session_id, patient_ref, note_text, safety_status,
                   fsm_final_state, processing_time_ms, created_at
            FROM clinical_sessions
            WHERE session_id = ?;
            """,
            [session_id],
        )
        row = cursor.fetchone()
        if not row:
            return None

        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    def get_safety_performance(self) -> list[dict[str, Any]]:
        """
        Query the v_safety_performance view to retrieve P50, P95, and P99 latencies.
        """
        cursor = self._conn.execute("SELECT * FROM v_safety_performance;")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def close(self) -> None:
        """Safely close the DuckDB connection."""
        self._conn.close()