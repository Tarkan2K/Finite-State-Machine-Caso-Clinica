"""
DuckDB analytical storage layer for clinical sessions, telemetry, patients, and local RxNorm concepts.
Provides high-performance columnar ingestion, in-memory patient lookup, and sub-millisecond RxNorm resolution.
"""

import json
from datetime import UTC, datetime
from typing import Any

import duckdb

from src.exceptions import StorageError
from src.types import ExtractedMedication, PatientAuditRecord, PatientRecord

CHILEAN_MOCK_PATIENTS: list[dict[str, Any]] = [
    {
        "rut": "12.345.678-5",
        "nombre": "Eleanor Vance Riquelme",
        "edad": 67,
        "ciudad": "Santiago",
        "enfermedades_cronicas": ["Diabetes Mellitus Tipo 2", "Insuficiencia Renal Moderada"],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": ["Metformina 850mg c/12h oral"],
    },
    {
        "rut": "18.901.234-K",
        "nombre": "Carlos Silva Araya",
        "edad": 24,
        "ciudad": "Concepción",
        "enfermedades_cronicas": [],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": [],
    },
    {
        "rut": "8.765.432-1",
        "nombre": "Arthur Pendelton Morales",
        "edad": 72,
        "ciudad": "Valparaíso",
        "enfermedades_cronicas": ["Trombosis Venosa Profunda", "Fibrilación Auricular"],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": ["Warfarina 5mg diario oral"],
    },
    {
        "rut": "15.892.431-2",
        "nombre": "Valentina Henríquez Soto",
        "edad": 45,
        "ciudad": "Temuco",
        "enfermedades_cronicas": ["Depresión Mayor Severa", "Fibromialgia"],
        "alergias": ["Penicilina"],
        "farmacos_actuales": ["Sertralina 50mg diario oral"],
    },
    {
        "rut": "11.456.789-3",
        "nombre": "Jorge Tapia Alarcón",
        "edad": 61,
        "ciudad": "Antofagasta",
        "enfermedades_cronicas": ["Artritis Reumatoide Activa", "Hipertensión Arterial"],
        "alergias": ["Sulfas"],
        "farmacos_actuales": ["Metotrexato 15mg semanal oral", "Enalapril 10mg c/12h oral"],
    },
    {
        "rut": "14.238.910-4",
        "nombre": "Camila Figueroa Rojas",
        "edad": 34,
        "ciudad": "Viña del Mar",
        "enfermedades_cronicas": ["Asma Bronquial Severa Persistente"],
        "alergias": ["AINEs (Aspirina e Ibuprofeno desencadenan broncoespasmo severo)"],
        "farmacos_actuales": ["Salbutamol 100mcg inhalador SOS", "Budesonida 200mcg c/12h"],
    },
    {
        "rut": "16.789.012-7",
        "nombre": "Mateo González Tapia",
        "edad": 29,
        "ciudad": "Rancagua",
        "enfermedades_cronicas": ["Dolor Lumbar Crónico Post-Traumático"],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": ["Paracetamol 500mg SOS"],
    },
    {
        "rut": "9.345.678-9",
        "nombre": "María José Castro Lagos",
        "edad": 74,
        "ciudad": "La Serena",
        "enfermedades_cronicas": ["Insuficiencia Cardíaca Congestiva NYHA II", "Fibrilación Auricular"],
        "alergias": ["Medio de Contraste Yodado (Shock Anafiláctico previo)"],
        "farmacos_actuales": ["Warfarina 4mg diario oral", "Furosemida 40mg diario oral"],
    },
    {
        "rut": "17.123.456-8",
        "nombre": "Rodrigo Valenzuela Pinto",
        "edad": 38,
        "ciudad": "Talca",
        "enfermedades_cronicas": ["Hipertensión Arterial Grado 1"],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": ["Losartán 50mg diario oral"],
    },
    {
        "rut": "13.567.890-1",
        "nombre": "Sofía Muñoz Sepúlveda",
        "edad": 53,
        "ciudad": "Puerto Montt",
        "enfermedades_cronicas": ["Diabetes Mellitus Tipo 2", "Neuropatía Diabética"],
        "alergias": ["Tramadol (Reacción Adversa: Náusea y Vértigo Intratable)"],
        "farmacos_actuales": ["Metformina 850mg c/12h oral", "Pregabalina 75mg nocturno"],
    },
    {
        "rut": "10.234.567-4",
        "nombre": "Pedro Pablo Morales Baeza",
        "edad": 69,
        "ciudad": "Iquique",
        "enfermedades_cronicas": ["EPOC Gold 2", "Cardiopatía Isquémica Crónica"],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": ["Aspirina 100mg diario oral", "Atorvastatina 20mg diario"],
    },
    {
        "rut": "19.456.789-0",
        "nombre": "Javiera Paz Arratia Núñez",
        "edad": 22,
        "ciudad": "Valdivia",
        "enfermedades_cronicas": ["Gastritis Crónica Erosiva", "Anemia Ferropénica"],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": ["Omeprazol 20mg ayunas oral"],
    },
    {
        "rut": "15.112.334-K",
        "nombre": "Gonzalo Ignacio Vergara Ríos",
        "edad": 41,
        "ciudad": "Chillán",
        "enfermedades_cronicas": ["Trastorno Afectivo Bipolar Tipo 1"],
        "alergias": ["Penicilina"],
        "farmacos_actuales": ["Carbonato de Litio 300mg c/8h oral", "Quetiapina 100mg nocturno"],
    },
    {
        "rut": "12.987.654-3",
        "nombre": "Francisca Andrea Díaz Fuentes",
        "edad": 58,
        "ciudad": "Osorno",
        "enfermedades_cronicas": ["Hipotiroidismo Primario", "Dislipidemia Mixta"],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": ["Levotiroxina 100mcg ayunas oral"],
    },
    {
        "rut": "16.445.566-7",
        "nombre": "Andrés Felipe Pizarro Godoy",
        "edad": 31,
        "ciudad": "Copiapó",
        "enfermedades_cronicas": ["Rinitis Alérgica Severa"],
        "alergias": ["Metamizol / Dipirona"],
        "farmacos_actuales": ["Levocetirizina 5mg diario oral"],
    },
    {
        "rut": "7.890.123-2",
        "nombre": "Héctor Hernán Navarrete Cruz",
        "edad": 81,
        "ciudad": "Punta Arenas",
        "enfermedades_cronicas": ["Demencia Senil Tipo Alzheimer", "Artrosis Severa de Cadera"],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": ["Donepezilo 10mg noche", "Paracetamol 1g c/8h oral"],
    },
    {
        "rut": "14.778.899-1",
        "nombre": "Daniela Paz Carrasco Leighton",
        "edad": 36,
        "ciudad": "Curicó",
        "enfermedades_cronicas": ["Migraña Crónica Frecuente"],
        "alergias": ["Codeína"],
        "farmacos_actuales": ["Propranolol 40mg c/12h oral"],
    },
    {
        "rut": "18.223.344-5",
        "nombre": "Sebastián Ignacio Palma Toro",
        "edad": 26,
        "ciudad": "Calama",
        "enfermedades_cronicas": [],
        "alergias": ["Ninguna conocida"],
        "farmacos_actuales": [],
    },
    {
        "rut": "13.889.900-6",
        "nombre": "Gloria del Carmen Urrutia Vidal",
        "edad": 63,
        "ciudad": "Los Ángeles",
        "enfermedades_cronicas": ["Insuficiencia Renal Crónica Terminal (Hemodiálisis)"],
        "alergias": ["AINEs"],
        "farmacos_actuales": ["Eritropoyetina 4000 UI semanal", "Calcitriol 0.25mcg diario"],
    },
    {
        "rut": "20.334.455-8",
        "nombre": "Benjamín Nicolás Herrera Bustos",
        "edad": 19,
        "ciudad": "Quillota",
        "enfermedades_cronicas": ["Asma Alérgica Intermitente"],
        "alergias": ["Aspirina (Ácido Acetilsalicílico)"],
        "farmacos_actuales": ["Salbutamol 100mcg inhalador SOS"],
    },
]

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
    Persists clinical sessions, extracted entities, patient registry, and local RxNorm concepts.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        try:
            self._conn = duckdb.connect(db_path)
            self._initialize_schema()
            self._seed_data()
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

            CREATE TABLE IF NOT EXISTS patients (
                rut VARCHAR PRIMARY KEY,
                nombre VARCHAR NOT NULL,
                edad INTEGER NOT NULL,
                ciudad VARCHAR NOT NULL,
                enfermedades_cronicas VARCHAR NOT NULL,
                alergias VARCHAR NOT NULL,
                farmacos_actuales VARCHAR NOT NULL
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

    def _seed_data(self) -> None:
        """Populate patients and local RxNorm concepts."""
        # Seed patients
        for p in CHILEAN_MOCK_PATIENTS:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO patients (
                    rut, nombre, edad, ciudad, enfermedades_cronicas, alergias, farmacos_actuales
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    p["rut"],
                    p["nombre"],
                    p["edad"],
                    p["ciudad"],
                    json.dumps(p["enfermedades_cronicas"], ensure_ascii=False),
                    json.dumps(p["alergias"], ensure_ascii=False),
                    json.dumps(p["farmacos_actuales"], ensure_ascii=False),
                ],
            )

        # Seed RxNorm concepts
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
        if not row:
            return None

        return {
            "rut": row[0],
            "nombre": row[1],
            "edad": row[2],
            "ciudad": row[3],
            "enfermedades_cronicas": json.loads(row[4]),
            "alergias": json.loads(row[5]),
            "farmacos_actuales": json.loads(row[6]),
        }

    def list_patients(self) -> list[dict[str, Any]]:
        """List all registered Chilean mock patients for the EHR directory."""
        cursor = self._conn.execute(
            "SELECT rut, nombre, edad, ciudad, enfermedades_cronicas, alergias, farmacos_actuales FROM patients ORDER BY nombre;"
        )
        rows = cursor.fetchall()
        return [
            {
                "rut": r[0],
                "nombre": r[1],
                "edad": r[2],
                "ciudad": r[3],
                "enfermedades_cronicas": json.loads(r[4]),
                "alergias": json.loads(r[5]),
                "farmacos_actuales": json.loads(r[6]),
            }
            for r in rows
        ]

    def resolve_rxnorm(self, raw_term: str) -> dict[str, Any] | None:
        """
        Sub-millisecond local in-process resolution of clinical term to RxNorm concept.
        Runs completely offline without external HTTP dependency.
        """
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
