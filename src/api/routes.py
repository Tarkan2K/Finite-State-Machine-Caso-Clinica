"""
FastAPI REST API routes for Aura Clinical runtime engine.
Exposes Chilean EHR patient lookups, clinical audit execution, session retrieval, and OLAP metrics.
Serves the ultra-lightweight Web2 / Windows XP bedside dashboard at root path.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse

from src.exceptions import EngineError, StorageError
from src.orchestrator import ClinicalOrchestrator
from src.store import ClinicalAuditStore
from src.types import ClinicalNote, PatientAuditRecord

# Global store reference for standalone app execution
_store: ClinicalAuditStore | None = None
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


def get_store() -> ClinicalAuditStore:
    """Dependency provider for persistent DuckDB audit store."""
    global _store
    if _store is None:
        _store = ClinicalAuditStore()
    return _store


def get_orchestrator(
    store: ClinicalAuditStore = Depends(get_store),
) -> ClinicalOrchestrator:
    """Dependency provider for clinical orchestrator bound to active store."""
    return ClinicalOrchestrator(store=store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    get_store()
    yield
    global _store
    if _store is not None:
        _store.close()
        _store = None


router = APIRouter(prefix="/api/v1", tags=["Clinical Safety Engine"])


@router.get(
    "/patients",
    response_model=list[dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="List Chilean Mock Patients",
    description="Retrieve all registered Chilean patients with RUT, chronic diseases, allergies, and current medications.",
)
async def list_patients(
    store: ClinicalAuditStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Retrieve full patient directory from DuckDB."""
    return store.list_patients()


@router.get(
    "/patients/{rut}",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Patient by Chilean RUT",
    description="Retrieve full EHR profile of a patient by their Chilean RUT.",
)
async def get_patient_by_rut(
    rut: str,
    store: ClinicalAuditStore = Depends(get_store),
) -> dict[str, Any]:
    """Retrieve individual patient file."""
    patient = store.get_patient(rut)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paciente con RUT '{rut}' no encontrado en el sistema.",
        )
    return patient


@router.post(
    "/clinical/audit",
    response_model=PatientAuditRecord,
    status_code=status.HTTP_200_OK,
    summary="Execute Clinical Safety Audit",
    description="Submit an unstructured clinical note for entity extraction, FSM-governed safety checks, and DuckDB logging.",
)
async def execute_clinical_audit(
    note: ClinicalNote,
    orchestrator: ClinicalOrchestrator = Depends(get_orchestrator),
) -> PatientAuditRecord:
    """Execute end-to-end clinical note safety audit."""
    try:
        record = await orchestrator.execute_audit(note=note)
        return record
    except EngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engine runtime failure: {exc}",
        ) from exc


@router.get(
    "/clinical/sessions/{session_id}",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Retrieve Session Audit Trail",
    description="Fetch a persisted clinical audit record by its unique session UUID.",
)
async def get_session_audit(
    session_id: str,
    store: ClinicalAuditStore = Depends(get_store),
) -> dict[str, Any]:
    """Retrieve persisted audit session from DuckDB."""
    try:
        session = store.get_session(session_id=session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Audit session '{session_id}' not found.",
            )
        return session
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage retrieval error: {exc}",
        ) from exc


@router.get(
    "/analytics/safety-metrics",
    response_model=list[dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Get Safety Latency Metrics (OLAP)",
    description="Retrieve real-time P50, P95, and P99 latency percentiles aggregated per tool via DuckDB.",
)
async def get_safety_metrics(
    store: ClinicalAuditStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Retrieve OLAP latency percentile metrics."""
    try:
        return store.get_safety_performance()
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics query error: {exc}",
        ) from exc


# Root FastAPI Application
app = FastAPI(
    title="Aura Clinical — FSM Agent Runtime API",
    description="Deterministic state-machine governed execution runtime for high-stakes clinical safety agents.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def serve_dashboard() -> HTMLResponse:
    """Serve the interactive Bedside Clinical Safety Web Portal."""
    if STATIC_INDEX.exists():
        return HTMLResponse(content=STATIC_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Aura Clinical Safety API is running.</h1>")


@app.get("/health", tags=["Monitoring"])
async def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "healthy", "engine": "fsm-agent-runtime"}
