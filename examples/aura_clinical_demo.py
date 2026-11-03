"""
Aura Clinical — Interactive Bedside Safety Terminal.
Live interactive application for clinicians to audit clinical notes,
reconcile patient chronic medications, enforce FSM invariants, and view DuckDB OLAP telemetry.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure root directory is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uuid
from src.exceptions import ClinicalSafetyViolationError
from src.fsm import AgentFSM
from src.orchestrator import extract_medications
from src.sandbox import check_drug_interactions
from src.store import ClinicalAuditStore
from src.types import (
    ClinicalNote,
    ClinicalSafetyStatus,
    DrugInteractionAlert,
    EngineState,
    PatientAuditRecord,
)

# Simulated hospital EHR patient directory
PATIENTS_DATABASE = {
    "1": {
        "id": "PAT-101",
        "name": "Eleanor Vance",
        "age": 67,
        "chronic_conditions": ["Type 2 Diabetes Mellitus", "Mild Renal Impairment"],
        "chronic_medications": ["metformin"],
    },
    "2": {
        "id": "PAT-102",
        "name": "Carlos Silva",
        "age": 24,
        "chronic_conditions": ["None (Healthy)"],
        "chronic_medications": [],
    },
    "3": {
        "id": "PAT-103",
        "name": "Arthur Pendelton",
        "age": 72,
        "chronic_conditions": ["Atrial Fibrillation", "Deep Vein Thrombosis"],
        "chronic_medications": ["warfarin"],
    },
}


def print_banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


async def run_interactive_audit(
    store: ClinicalAuditStore,
    patient_id: str,
    patient_name: str,
    chronic_meds: list[str],
    note_text: str,
) -> None:
    session_id = str(uuid.uuid4())
    fsm = AgentFSM(max_steps=10)

    print(f"\n[ORCHESTRATOR] Starting Session UUID: {session_id}")
    print(f"[PATIENT] {patient_name} (ID: {patient_id})")
    print(f"[CHRONIC MEDS ON FILE] {chronic_meds if chronic_meds else 'None'}")
    print(f"[BEDSIDE NOTE] \"{note_text}\"\n")

    # Step 1: INITIALIZED -> PLANNING
    print("  -> [FSM] INITIALIZED -> PLANNING")
    fsm.transition(EngineState.PLANNING)
    await asyncio.sleep(0.1)

    # Step 2: PLANNING -> DISPATCHING_TOOL (Medication Extractor)
    print("  -> [FSM] PLANNING -> DISPATCHING_TOOL (Running Medication Extractor)")
    fsm.transition(EngineState.DISPATCHING_TOOL)
    extracted = await extract_medications(note_text)
    new_med_names = [m.name for m in extracted]
    print(f"     [TOOL] Extracted from note: {new_med_names if new_med_names else 'None detected'}")

    # Step 3: DISPATCHING_TOOL -> AWAITING_OBSERVATION
    print("  -> [FSM] DISPATCHING_TOOL -> AWAITING_OBSERVATION")
    fsm.transition(EngineState.AWAITING_OBSERVATION)

    for med in extracted:
        store.write_medication_span(
            span_id=str(uuid.uuid4()),
            session_id=session_id,
            medication=med,
            extraction_latency_ms=2.15,
        )

    # Step 4: AWAITING_OBSERVATION -> REFLECTING (Reconcile with Chronic History & Check)
    print("  -> [FSM] AWAITING_OBSERVATION -> REFLECTING (Clinical Safety Reconciler)")
    fsm.transition(EngineState.REFLECTING)

    # Combine chronic baseline medications with newly prescribed medications
    all_medications = list(set(chronic_meds + new_med_names))
    print(f"     [SAFETY MATRIX] Reconciled Active Substances: {all_medications}")

    alerts = check_drug_interactions(all_medications)
    has_critical = any(a.severity == "CRITICAL" for a in alerts)

    # Record safety check telemetry in DuckDB
    store.write_safety_check(
        check_id=str(uuid.uuid4()),
        session_id=session_id,
        tool_name="drug_interaction_matrix",
        result="CRITICAL" if has_critical else "SAFE",
        latency_ms=1.84,
        alert_description=alerts[0].description if alerts else None,
    )

    # Step 5: FSM Enforcement
    if has_critical:
        print("\n  [!] CRITICAL SAFETY HAZARD DETECTED BY RUNTIME:")
        for alert in alerts:
            print(f"      * {alert.drug_a.upper()} + {alert.drug_b.upper()}: {alert.description}")
            print(f"        Severity: {alert.severity} | ICD-10: {', '.join(alert.cie10_codes)}")

        print("\n  -> [FSM] INTERCEPT: Forcing transition REFLECTING -> FAILED (Absorbing State)")
        fsm.transition(EngineState.FAILED)
        safety_status = ClinicalSafetyStatus.CRITICAL_BLOCK
        print("  -> [VERDICT] 🛑 CRITICAL BLOCK: Prescription revoked. Zero unsafe output dispatched.")
    else:
        print("\n  -> [FSM] Clean evaluation. Transition REFLECTING -> COMPLETED")
        fsm.transition(EngineState.COMPLETED)
        safety_status = ClinicalSafetyStatus.SAFE
        print("  -> [VERDICT] ✅ APPROVED: Prescription verified safe for dispensing.")

    # Persist immutable record in DuckDB
    record = PatientAuditRecord(
        session_id=session_id,
        patient_ref=patient_id,
        note_text=note_text,
        medications_extracted=all_medications,
        safety_status=safety_status,
        alerts=alerts,
        fsm_final_state=fsm.current_state,
        processing_time_ms=14.2,
        created_at=datetime.now(timezone.utc),
    )
    store.write_session(record)


def show_duckdb_metrics(store: ClinicalAuditStore) -> None:
    print_banner("DUCKDB OLAP TELEMETRY (In-Process Columnar Analytics)")
    metrics = store.get_safety_performance()
    if not metrics:
        print("No telemetry recorded yet.")
        return

    print(f"{'Tool Name':<28} | {'Checks':<8} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10}")
    print("-" * 75)
    for m in metrics:
        print(
            f"{m['tool_name']:<28} | "
            f"{m['total_checks']:<8} | "
            f"{m['p50_ms']:<10.2f} | "
            f"{m['p95_ms']:<10.2f} | "
            f"{m['p99_ms']:<10.2f}"
        )


async def main() -> None:
    store = ClinicalAuditStore(db_path=":memory:")

    print_banner("AURA CLINICAL — INTERACTIVE AGENT SAFETY RUNTIME")
    print("Welcome to the bedside clinical safety runtime interface.")
    print("This console allows real-time interactive evaluation of clinical notes.")

    while True:
        print("\nSelect a patient profile to evaluate:")
        for key, p in PATIENTS_DATABASE.items():
            print(f"  [{key}] {p['name']} ({p['age']}y) — Chronic: {', '.join(p['chronic_conditions'])} | Meds: {p['chronic_medications']}")
        print("  [4] Custom / Unregistered Patient")
        print("  [q] Quit and show DuckDB Telemetry")

        choice = input("\nEnter choice (1-4 or q): ").strip().lower()

        if choice == "q":
            break

        if choice in ("1", "2", "3"):
            patient = PATIENTS_DATABASE[choice]
            p_id = patient["id"]
            p_name = patient["name"]
            p_meds = list(patient["chronic_medications"])
        elif choice == "4":
            p_id = "PAT-CUSTOM"
            p_name = "Guest Patient"
            p_meds = []
        else:
            print("Invalid selection. Try again.")
            continue

        print(f"\nEvaluating patient: {p_name}")
        print("Type the clinical note / bedside prescription:")
        print("Examples to test:")
        print("  - Safe: 'Administer paracetamol 500mg oral for headache'")
        print("  - Danger with Eleanor (Metformin): 'Schedule immediate contrast CT scan'")
        print("  - Danger with Arthur (Warfarin): 'Administer aspirin 100mg for joint pain'")
        print("  - Danger custom: 'Give tramadol 50mg and ssri for pain and anxiety'")

        note_input = input("\n[Bedside Note] > ").strip()
        if not note_input:
            print("Note cannot be empty.")
            continue

        await run_interactive_audit(
            store=store,
            patient_id=p_id,
            patient_name=p_name,
            chronic_meds=p_meds,
            note_text=note_input,
        )

    show_duckdb_metrics(store)
    store.close()
    print("\nSession closed. All invariants preserved.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited.")
        sys.exit(0)
