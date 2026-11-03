"""
DuckDB analytical storage layer for clinical sessions, FSM checkpoints, telemetry, and patients.
Reads patient data directly from external CSV file (Base_de_datos_ficticia/pacientes.csv).
"""

import json
from datetime import timezone, datetime
from pathlib import Path
from typing import Any

import duckdb

from src.exceptions import StorageError
from src.types import ExtractedMedication, PatientAuditRecord

BASE_DIR = Path(__file__).resolve().parent.parent
PATIENTS_CSV = BASE_DIR / "Base_de_datos_ficticia" / "pacientes.csv"
DEFAULT_DB_PATH = str(BASE_DIR / "Base_de_datos_ficticia" / "clinical_audit.duckdb")


class ClinicalAuditStore:
    """
    Embedded analytical audit store powered by DuckDB.
    Persists clinical sessions, FSM state checkpoints, telemetry spans, and CSV patient records.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        try:
            self._conn = duckdb.connect(db_path)
            self._initialize_schema()
            self._load_patients_from_csv()
        except Exception as exc:
            raise StorageError(f"Failed to initialize DuckDB storage: {exc}") from exc

    def _initialize_schema(self) -> None:
        """Create relational tables and analytical OLAP views."""
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

            CREATE TABLE IF NOT EXISTS fsm_checkpoints (
                session_id VARCHAR PRIMARY KEY,
                current_state VARCHAR NOT NULL,
                step_count INTEGER NOT NULL,
                checkpoint_json TEXT NOT NULL,
                updated_at VARCHAR NOT NULL
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

    def _load_patients_from_csv(self) -> None:
        """Load patient table directly from Base_de_datos_ficticia/pacientes.csv."""
        if not PATIENTS_CSV.exists():
            raise StorageError(f"Patient database file not found: {PATIENTS_CSV}")

        csv_path_escaped = str(PATIENTS_CSV).replace("\\", "/")
        self._conn.execute(
            f"""
            CREATE OR REPLACE TABLE patients AS
            SELECT
                TRIM(rut) AS rut,
                TRIM(nombre) AS nombre,
                CAST(edad AS INTEGER) AS edad,
                TRIM(ciudad) AS ciudad,
                TRIM(enfermedades_cronicas) AS enfermedades_cronicas,
                TRIM(alergias) AS alergias,
                TRIM(farmacos_actuales) AS farmacos_actuales
            FROM read_csv_auto('{csv_path_escaped}', header=true);
            """
        )

    def _format_patient_row(self, row: tuple) -> dict[str, Any]:
        """Convert raw DuckDB row into structured clinical dictionary."""
        def split_entries(val: str) -> list[str]:
            if not val or val.lower() in ("ninguna", "ninguno", "ninguna conocida"):
                return []
            return [s.strip() for s in val.split(";") if s.strip()]

        return {
            "rut": row[0],
            "nombre": row[1],
            "edad": row[2],
            "ciudad": row[3],
            "enfermedades_cronicas": split_entries(row[4]),
            "alergias": split_entries(row[5]),
            "farmacos_actuales": split_entries(row[6]),
        }

    def get_patient(self, rut: str) -> dict[str, Any] | None:
        """Fetch patient clinical profile by Chilean RUT."""
        clean_rut = rut.strip().upper()
        cursor = self._conn.execute(
            """
            SELECT rut, nombre, edad, ciudad, enfermedades_cronicas, alergias, farmacos_actuales
            FROM patients
            WHERE UPPER(rut) = ?;
            """,
            [clean_rut],
        )
        row = cursor.fetchone()
        return self._format_patient_row(row) if row else None

    def list_patients(self) -> list[dict[str, Any]]:
        """List all patients from the CSV database."""
        cursor = self._conn.execute(
            "SELECT rut, nombre, edad, ciudad, enfermedades_cronicas, alergias, farmacos_actuales FROM patients ORDER BY nombre;"
        )
        return [self._format_patient_row(r) for r in cursor.fetchall()]

    def save_checkpoint(self, session_id: str, checkpoint: dict[str, Any]) -> None:
        """Persist or update an FSM state checkpoint in DuckDB."""
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO fsm_checkpoints (
                    session_id, current_state, step_count, checkpoint_json, updated_at
                ) VALUES (?, ?, ?, ?, ?);
                """,
                [
                    session_id,
                    str(checkpoint["current_state"]),
                    int(checkpoint["step_count"]),
                    json.dumps(checkpoint),
                    datetime.now(timezone.utc).isoformat(),
                ],
            )
        except Exception as exc:
            raise StorageError(f"Failed to persist FSM checkpoint: {exc}") from exc

    def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Load a persisted FSM checkpoint by session_id."""
        cursor = self._conn.execute(
            "SELECT checkpoint_json FROM fsm_checkpoints WHERE session_id = ?;",
            [session_id],
        )
        row = cursor.fetchone()
        return json.loads(row[0]) if row else None

    def write_session(self, record: PatientAuditRecord) -> None:
        """Persist a finalized patient audit record."""
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
                    datetime.now(timezone.utc).isoformat(),
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
                    datetime.now(timezone.utc).isoformat(),
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
        """Query the v_safety_performance view to retrieve P50, P95, and P99 latencies."""
        cursor = self._conn.execute("SELECT * FROM v_safety_performance;")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def close(self) -> None:
        """Safely close the DuckDB connection."""
        self._conn.close()
