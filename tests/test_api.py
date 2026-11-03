"""
API integration tests for FastAPI REST routes.
Validates HTTP request/response contracts, status codes, Pydantic validation, and DuckDB retrieval.
"""

from fastapi.testclient import TestClient


def test_health_check_endpoint(client: TestClient) -> None:
    """GET /health must return 200 OK with engine status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "engine": "fsm-agent-runtime"}


def test_dashboard_endpoint(client: TestClient) -> None:
    """GET / must return HTML dashboard page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Sistema Clínico" in response.text


def test_execute_audit_safe_endpoint(client: TestClient) -> None:
    """POST /api/v1/clinical/audit must return 200 with SAFE verdict for clean notes."""
    payload = {
        "note_id": "API-TEST-001",
        "patient_ref": "18.901.234-K",
        "raw_text": "Patient has mild fever. Administer paracetamol 500mg.",
    }
    response = client.post("/api/v1/clinical/audit", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["safety_status"] == "SAFE"
    assert data["fsm_final_state"] == "COMPLETED"
    assert data["patient_ref"] == "18.901.234-K"
    assert "session_id" in data


def test_execute_audit_critical_block_endpoint(client: TestClient) -> None:
    """POST /api/v1/clinical/audit must return CRITICAL_BLOCK on dangerous drug combinations."""
    payload = {
        "note_id": "API-TEST-CRIT",
        "patient_ref": "12.345.678-5",
        "raw_text": "Administer metformin 500mg and schedule contrast tomography scan.",
    }
    response = client.post("/api/v1/clinical/audit", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["safety_status"] == "CRITICAL_BLOCK"
    assert data["fsm_final_state"] == "FAILED"
    assert len(data["alerts"]) >= 1
    assert data["alerts"][0]["severity"] == "CRITICAL"


def test_execute_audit_invalid_schema_returns_422(client: TestClient) -> None:
    """Malformed request payload missing mandatory fields must be rejected with 422."""
    payload = {
        # missing patient_ref and raw_text
        "note_id": "INCOMPLETE-PAYLOAD"
    }
    response = client.post("/api/v1/clinical/audit", json=payload)
    assert response.status_code == 422


def test_get_session_audit_flow(client: TestClient) -> None:
    """Session created via audit must be retrievable via GET /clinical/sessions/{id}."""
    # 1. Trigger an audit
    payload = {
        "note_id": "API-SESSION-001",
        "patient_ref": "18.901.234-K",
        "raw_text": "Administer aspirin 100mg oral.",
    }
    create_res = client.post("/api/v1/clinical/audit", json=payload)
    assert create_res.status_code == 200
    session_id = create_res.json()["session_id"]

    # 2. Query persisted session
    get_res = client.get(f"/api/v1/clinical/sessions/{session_id}")
    assert get_res.status_code == 200
    assert get_res.json()["session_id"] == session_id
    assert get_res.json()["patient_ref"] == "18.901.234-K"


def test_get_nonexistent_session_returns_404(client: TestClient) -> None:
    """Querying an unknown session_id must return 404 Not Found."""
    response = client.get("/api/v1/clinical/sessions/unknown-session-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_safety_metrics_endpoint(client: TestClient) -> None:
    """GET /api/v1/analytics/safety-metrics must return 200 with list of metrics."""
    # Run an audit to generate data
    payload = {
        "note_id": "API-METRIC-01",
        "patient_ref": "12.345.678-5",
        "raw_text": "Metformin 500mg and contrast study.",
    }
    client.post("/api/v1/clinical/audit", json=payload)

    metrics_res = client.get("/api/v1/analytics/safety-metrics")
    assert metrics_res.status_code == 200
    assert isinstance(metrics_res.json(), list)


def test_list_chilean_patients_endpoint(client: TestClient) -> None:
    """GET /api/v1/patients must return all seeded Chilean mock patients."""
    response = client.get("/api/v1/patients")
    assert response.status_code == 200
    patients = response.json()
    assert len(patients) >= 20
    ruts = [p["rut"] for p in patients]
    assert "12.345.678-5" in ruts
    assert "18.901.234-K" in ruts


def test_get_patient_by_rut_endpoint(client: TestClient) -> None:
    """GET /api/v1/patients/{rut} must return patient dossier with chronic meds and allergies."""
    response = client.get("/api/v1/patients/12.345.678-5")
    assert response.status_code == 200
    data = response.json()
    assert data["nombre"] == "Eleanor Vance Riquelme"
    assert data["edad"] == 67
    assert "Diabetes Mellitus Tipo 2" in data["enfermedades_cronicas"]
    assert any("Metformina" in m for m in data["farmacos_actuales"])


def test_get_nonexistent_patient_returns_404(client: TestClient) -> None:
    """Querying an unregistered RUT must return 404."""
    response = client.get("/api/v1/patients/99.999.999-9")
    assert response.status_code == 404
    assert "no encontrado" in response.json()["detail"].lower()

