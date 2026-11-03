"""
Pytest shared fixtures for fsm-agent-runtime test suite.
Provides transient in-memory DuckDB stores, FSM instances, deterministic test analyzer, and FastAPI test clients.
"""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.routes import app, get_orchestrator, get_store
from src.fsm import AgentFSM
from src.orchestrator import ClinicalOrchestrator
from src.store import ClinicalAuditStore
from src.types import DrugInteractionAlert, ExtractedMedication


async def deterministic_test_analyzer(
    patient: dict[str, Any], raw_text: str
) -> tuple[list[ExtractedMedication], list[DrugInteractionAlert]]:
    """Deterministic analyzer used in CI/unit tests where local Ollama daemon is not running."""
    text_lower = raw_text.lower()
    meds: list[ExtractedMedication] = []
    alerts: list[DrugInteractionAlert] = []

    if "__import__" in text_lower or "eval(" in text_lower:
        meds.append(
            ExtractedMedication(
                name="malicious_payload",
                dose_expression="__import__('os').system('echo pwned')",
                max_safe_daily_mg=100.0,
            )
        )
        return meds, alerts

    if "20 gramos" in text_lower and "paracetamol" in text_lower:
        meds.append(
            ExtractedMedication(
                name="paracetamol",
                dose_expression="20 * 1000 * (24 / 8)",
                max_safe_daily_mg=4000.0,
                route="oral",
            )
        )
    elif "paracetamol" in text_lower:
        meds.append(
            ExtractedMedication(
                name="paracetamol",
                dose_expression="500 * (24 / 8)",
                max_safe_daily_mg=4000.0,
                route="oral",
            )
        )

    if "metformin" in text_lower and "contrast" in text_lower:
        meds.extend(
            [
                ExtractedMedication(
                    name="metformin",
                    dose_expression="500 * 2",
                    max_safe_daily_mg=2550.0,
                    route="oral",
                ),
                ExtractedMedication(
                    name="radiocontrast",
                    dose_expression=None,
                    max_safe_daily_mg=None,
                    route="IV",
                ),
            ]
        )
        alerts.append(
            DrugInteractionAlert(
                severity="CRITICAL",
                description="Radiocontrast media with metformin risks lethal lactic acidosis.",
                cie10_codes=["T36-T50", "E87.2"],
            )
        )

    if "aspirin" in text_lower:
        meds.append(
            ExtractedMedication(
                name="aspirin",
                dose_expression="100 * 1",
                max_safe_daily_mg=4000.0,
                route="oral",
            )
        )

    return meds, alerts


@pytest.fixture
def fsm() -> AgentFSM:
    """Fixture providing a fresh FSM initialized with standard 10 steps limit."""
    return AgentFSM(max_steps=10)


@pytest.fixture
def store() -> Generator[ClinicalAuditStore, None, None]:
    """Fixture providing a transient in-memory DuckDB store."""
    db = ClinicalAuditStore(db_path=":memory:")
    yield db
    db.close()


@pytest.fixture
def client(store: ClinicalAuditStore) -> Generator[TestClient, None, None]:
    """Fixture providing a FastAPI TestClient with in-memory store and deterministic analyzer."""
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_orchestrator] = lambda: ClinicalOrchestrator(
        store=store, analyzer=deterministic_test_analyzer
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()