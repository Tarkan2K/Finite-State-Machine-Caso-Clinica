"""
Aura Clinical — Interactive Bedside Safety Terminal.
Live interactive console for clinicians to audit clinical notes using
the unified ClinicalOrchestrator (DuckDB EHR + Local LLM + AST Sandbox + FSM).
"""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestrator import ClinicalOrchestrator
from src.store import ClinicalAuditStore
from src.types import ClinicalNote, ClinicalSafetyStatus


def print_banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


async def main() -> None:
    store = ClinicalAuditStore()
    orchestrator = ClinicalOrchestrator(store=store)

    print_banner("AURA CLINICAL — INTERACTIVE AGENT SAFETY RUNTIME")
    patients = store.list_patients()

    while True:
        print("\nPacientes registrados (primeros 5):")
        for idx, p in enumerate(patients[:5], start=1):
            print(f"  [{idx}] {p['rut']} — {p['nombre']} ({p['edad']} años)")
        print("  [q] Salir y mostrar telemetría DuckDB")

        rut_input = input("\nIngrese RUT del paciente (o 'q' para salir): ").strip()
        if rut_input.lower() == "q":
            break

        patient = store.get_patient(rut_input)
        if not patient:
            print(f"Paciente con RUT '{rut_input}' no encontrado.")
            continue

        print(f"\nEvaluando a: {patient['nombre']} ({patient['edad']} años)")
        print(f"  Patologías: {', '.join(patient['enfermedades_cronicas']) or 'Ninguna'}")
        print(f"  Alergias: {', '.join(patient['alergias']) or 'Ninguna'}")
        print(f"  Fármacos actuales: {', '.join(patient['farmacos_actuales']) or 'Ninguno'}")

        note_input = input("\n[Indicación Médica] > ").strip()
        if not note_input:
            continue

        note = ClinicalNote(
            note_id="CLI-AUDIT",
            patient_ref=patient["rut"],
            raw_text=note_input,
        )
        record = await orchestrator.execute_audit(note=note)

        if record.safety_status == ClinicalSafetyStatus.CRITICAL_BLOCK:
            print(f"\n  [!] BLOQUEO CRÍTICO — Estado FSM: {record.fsm_final_state.value}")
            for alert in record.alerts:
                print(f"      * [{alert.severity}] {alert.description}")
                print(f"        CIE-10: {', '.join(alert.cie10_codes)}")
        else:
            print(f"\n  [OK] PRESCRIPCIÓN SEGURA — Estado FSM: {record.fsm_final_state.value}")

    store.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
