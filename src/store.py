"""
DuckDB analytical storage layer for clinical sessions, telemetry, patients, and local RxNorm concepts.
Reads patient data directly from external CSV file (Base_de_datos_ficticia/pacientes.csv).
Zero patient data hardcoded in Python.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from src.exceptions import StorageError
from src.types import ExtractedMedication, PatientAuditRecord

BASE_DIR = Path(__file__).resolve().parent.parent
PATIENTS_CSV = BASE_DIR / "Base_de_datos_ficticia" / "pacientes.csv"

RXNORM_SEED_CONCEPTS: list[dict[str, Any]] = [
    # Metformin (RxCUI 6809)
    {"rxcui": 6809, "term_name": "metformin", "canonical_name": "metformin", "category": "antidiabetic"},
    {"rxcui": 6809, "term_name": "metformina", "canonical_name": "metformin", "category": "antidiabetic"},
    {"rxcui": 6809, "term_name": "glifage", "canonical_name": "metformin", "category": "antidiabetic"},
    {"rxcui": 6809, "term_name": "hipoglucin", "canonical_name": "metformin", "category": "antidiabetic"},
    # Radiocontrast Media (RxCUI 376261)
    {"rxcui": 376261, "term_name": "radiocontrast", "canonical_name": "radiocontrast", "category": "diagnostic"},
    {"rxcui": 376261, "term_name": "contrast", "canonical_name": "radiocontrast", "category": "diagnostic"},
    {"rxcui": 376261, "term_name": "contraste", "canonical_name": "radiocontrast", "category": "diagnostic"},
    {"rxcui": 376261, "term_name": "medio de contraste", "canonical_name": "radiocontrast", "category": "diagnostic"},
    {"rxcui": 376261, "term_name": "contrast study", "canonical_name": "radiocontrast", "category": "diagnostic"},
    {"rxcui": 376261, "term_name": "contraste yodado", "canonical_name": "radiocontrast", "category": "diagnostic"},
    # Warfarin (RxCUI 11289)
    {"rxcui": 11289, "term_name": "warfarin", "canonical_name": "warfarin", "category": "anticoagulant"},
    {"rxcui": 11289, "term_name": "warfarina", "canonical_name": "warfarin", "category": "anticoagulant"},
    {"rxcui": 11289, "term_name": "coumadin", "canonical_name": "warfarin", "category": "anticoagulant"},
    # Aspirin (RxCUI 1191)
    {"rxcui": 1191, "term_name": "aspirin", "canonical_name": "aspirin", "category": "antiplatelet_nsaid"},
    {"rxcui": 1191, "term_name": "aspirina", "canonical_name": "aspirin", "category": "antiplatelet_nsaid"},
    {"rxcui": 1191, "term_name": "cardioaspirina", "canonical_name": "aspirin", "category": "antiplatelet_nsaid"},
    {"rxcui": 1191, "term_name": "acido acetilsalicilico", "canonical_name": "aspirin", "category": "antiplatelet_nsaid"},
    {"rxcui": 1191, "term_name": "ácido acetilsalicílico", "canonical_name": "aspirin", "category": "antiplatelet_nsaid"},
    # Paracetamol / Acetaminophen (RxCUI 161)
    {"rxcui": 161, "term_name": "paracetamol", "canonical_name": "paracetamol", "category": "analgesic"},
    {"rxcui": 161, "term_name": "acetaminophen", "canonical_name": "paracetamol", "category": "analgesic"},
    {"rxcui": 161, "term_name": "acetaminofen", "canonical_name": "paracetamol", "category": "analgesic"},
    {"rxcui": 161, "term_name": "acetaminofén", "canonical_name": "paracetamol", "category": "analgesic"},
    {"rxcui": 161, "term_name": "tylenol", "canonical_name": "paracetamol", "category": "analgesic"},
    {"rxcui": 161, "term_name": "kitadol", "canonical_name": "paracetamol", "category": "analgesic"},
    {"rxcui": 161, "term_name": "panadol", "canonical_name": "paracetamol", "category": "analgesic"},
    {"rxcui": 161, "term_name": "tempra", "canonical_name": "paracetamol", "category": "analgesic"},
    # Tramadol (RxCUI 10689)
    {"rxcui": 10689, "term_name": "tramadol", "canonical_name": "tramadol", "category": "opioid"},
    {"rxcui": 10689, "term_name": "tramal", "canonical_name": "tramadol", "category": "opioid"},
    {"rxcui": 10689, "term_name": "minitran", "canonical_name": "tramadol", "category": "opioid"},
    # SSRIs (RxCUI 36437)
    {"rxcui": 36437, "term_name": "ssri", "canonical_name": "ssri", "category": "antidepressant"},
    {"rxcui": 36437, "term_name": "isrs", "canonical_name": "ssri", "category": "antidepressant"},
    {"rxcui": 36437, "term_name": "sertralina", "canonical_name": "ssri", "category": "antidepressant"},
    {"rxcui": 36437, "term_name": "sertraline", "canonical_name": "ssri", "category": "antidepressant"},
    {"rxcui": 36437, "term_name": "fluoxetina", "canonical_name": "ssri", "category": "antidepressant"},
    {"rxcui": 36437, "term_name": "fluoxetine", "canonical_name": "ssri", "category": "antidepressant"},
    # Methotrexate (RxCUI 6851)
    {"rxcui": 6851, "term_name": "methotrexate", "canonical_name": "methotrexate", "category": "immunosuppressive"},
    {"rxcui": 6851, "term_name": "metotrexato", "canonical_name": "methotrexate", "category": "immunosuppressive"},
    # NSAIDs (RxCUI 5640)
    {"rxcui": 5640, "term_name": "nsaid", "canonical_name": "nsaid", "category": "nsaid"},
    {"rxcui": 5640, "term_name": "aine", "canonical_name": "nsaid", "category": "nsaid"},
    {"rxcui": 5640, "term_name": "aines", "canonical_name": "nsaid", "category": "nsaid"},
    {"rxcui": 5640, "term_name": "ibuprofen", "canonical_name": "ibuprofen", "category": "nsaid"},
    {"rxcui": 5640, "term_name": "ibuprofeno", "canonical_name": "ibuprofen", "category": "nsaid"},
    {"rxcui": 5640, "term_name": "advil", "canonical_name": "ibuprofen", "category": "nsaid"},
    {"rxcui": 5640, "term_name": "motrin", "canonical_name": "ibuprofen", "category": "nsaid"},
    # Cocaine (RxCUI 2670)
    {"rxcui": 2670, "term_name": "cocaine", "canonical_name": "cocaine", "category": "illicit_stimulant"},
    {"rxcui": 2670, "term_name": "cocaina", "canonical_name": "cocaine", "category": "illicit_stimulant"},
    {"rxcui": 2670, "term_name": "cocaína", "canonical_name": "cocaine", "category": "illicit_stimulant"},
    # Fentanyl (RxCUI 4337)
    {"rxcui": 4337, "term_name": "fentanyl", "canonical_name": "fentanyl", "category": "potent_opioid"},
    {"rxcui": 4337, "term_name": "fentanilo", "canonical_name": "fentanyl", "category": "potent_opioid"},
    {"rxcui": 4337, "term_name": "durogesic", "canonical_name": "fentanyl", "category": "potent_opioid"},
]


class ClinicalAuditStore:
    """
    Embedded analytical audit store powered by DuckDB.
    Persists clinical sessions, telemetry, and dynamically loads patients from Base_de_datos_ficticia/pacientes.csv.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        try:
            self._conn = duckdb.connect(db_path)
            self._initialize_schema()
            self._load_patients_from_csv()
            self._seed_rxnorm_concepts()
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

            CREATE TABLE IF NOT EXISTS rxnorm_concepts (
                term_name VARCHAR PRIMARY KEY,
                rxcui INTEGER NOT NULL,
                canonical_name VARCHAR NOT NULL,
                category VARCHAR NOT NULL
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
        """Load or reload patient table directly from Base_de_datos_ficticia/pacientes.csv."""
        if not PATIENTS_CSV.exists():
            raise StorageError(f"Patient database file not found: {PATIENTS_CSV}")

        # DuckDB creates the table directly from the CSV file
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

    def _seed_rxnorm_concepts(self) -> None:
        """Populate local RxNorm concepts."""
        for item in RXNORM_SEED_CONCEPTS:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO rxnorm_concepts (
                    term_name, rxcui, canonical_name, category
                ) VALUES (?, ?, ?, ?);
                """,
                [
                    item["term_name"].lower().strip(),
                    item["rxcui"],
                    item["canonical_name"].lower().strip(),
                    item["category"],
                ],
            )

    def _format_patient_row(self, row: tuple) -> dict[str, Any]:
        """Convert raw DuckDB row into structured clinical dictionary."""
        def split_entries(val: str) -> list[str]:
            if not val or val.lower() == "ninguna" or val.lower() == "ninguno":
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
        """Fetch patient clinical profile by Chilean RUT directly from the external table."""
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
        if not row:
            return None
        return self._format_patient_row(row)

    def list_patients(self) -> list[dict[str, Any]]:
        """List all patients from the CSV database."""
        cursor = self._conn.execute(
            "SELECT rut, nombre, edad, ciudad, enfermedades_cronicas, alergias, farmacos_actuales FROM patients ORDER BY nombre;"
        )
        rows = cursor.fetchall()
        return [self._format_patient_row(r) for r in rows]

    def resolve_rxnorm(self, raw_term: str) -> dict[str, Any] | None:
        """Sub-millisecond local resolution of clinical term to RxNorm concept."""
        clean_term = raw_term.lower().strip()
        cursor = self._conn.execute(
            """
            SELECT rxcui, canonical_name, category
            FROM rxnorm_concepts
            WHERE term_name = ?
            LIMIT 1;
            """,
            [clean_term],
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "rxcui": row[0],
            "canonical_name": row[1],
            "category": row[2],
        }

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
        """Query the v_safety_performance view to retrieve P50, P95, and P99 latencies."""
        cursor = self._conn.execute("SELECT * FROM v_safety_performance;")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    def close(self) -> None:
        """Safely close the DuckDB connection."""
        self._conn.close()
