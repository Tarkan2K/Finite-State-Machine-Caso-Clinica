"""
FastAPI REST API routes for Aura Clinical runtime engine.
Exposes clinical audit execution, session audit trail retrieval, and OLAP metrics.
"""

from contextlib import asynccontextmanager
from typing import Any
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from src.exceptions import EngineError, StorageError
from src.orchestrator import ClinicalOrchestrator
from src.store import ClinicalAuditStore
from src.types import ClinicalNote, PatientAuditRecord

# Global store reference for standalone app execution
_store: ClinicalAuditStore | None = None


def get_store() -> ClinicalAuditStore:
    """Dependency provider for DuckDB audit store."""
    global _store
    if _store is None:
        _store = ClinicalAuditStore(db_path=":memory:")
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
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health", tags=["Monitoring"])
async def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "healthy", "engine": "fsm-agent-runtime"}