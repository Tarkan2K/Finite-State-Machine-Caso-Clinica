"""
Pytest shared fixtures for fsm-agent-runtime test suite.
Provides transient in-memory DuckDB stores, FSM instances, and FastAPI test clients.
"""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from src.api.routes import app, get_store
from src.fsm import AgentFSM
from src.store import ClinicalAuditStore


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
    """Fixture providing a FastAPI TestClient with overridden in-memory store."""
    app.dependency_overrides[get_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()