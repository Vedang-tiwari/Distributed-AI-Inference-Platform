import pytest
from fastapi.testclient import TestClient
from api.main import app
from core.db.session import init_db

client = TestClient(app)
MASTER_KEY = "ai_live_master_key_999999"
AUTH_HEADER = {"Authorization": f"Bearer {MASTER_KEY}"}


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Distributed AI Inference Platform"
    assert data["status"] == "operational"


def test_health_and_ready():
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    ready_resp = client.get("/ready")
    assert ready_resp.status_code == 200
    assert ready_resp.json()["ready"] is True


def test_authentication_required():
    # Without header
    res = client.get("/v1/models")
    assert res.status_code == 401

    # With invalid key
    res = client.get("/v1/models", headers={"Authorization": "Bearer invalid_key"})
    assert res.status_code == 401

    # With valid master key
    res = client.get("/v1/models", headers=AUTH_HEADER)
    assert res.status_code == 200
    assert "models" in res.json()


def test_model_listing_and_versions():
    # List models
    res = client.get("/v1/models", headers=AUTH_HEADER)
    assert res.status_code == 200
    models = res.json()["models"]
    model_names = [m["name"] for m in models]
    assert "resnet" in model_names

    # Versions
    res_versions = client.get("/v1/models/resnet/versions", headers=AUTH_HEADER)
    assert res_versions.status_code == 200
    v_data = res_versions.json()
    assert v_data["model"] == "resnet"
    assert len(v_data["versions"]) >= 1


def test_model_registration():
    payload = {
        "name": "sentiment-analysis",
        "version": "v1",
        "framework": "pytorch",
        "device": "cpu",
        "artifact_uri": "s3://models/sentiment/v1/model.pt",
        "description": "DistilBERT sentiment analysis classifier",
    }
    res = client.post("/v1/models/register", json=payload, headers=AUTH_HEADER)
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # Verify present
    res_list = client.get("/v1/models", headers=AUTH_HEADER)
    names = [m["name"] for m in res_list.json()["models"]]
    assert "sentiment-analysis" in names


def test_canary_routing_configuration():
    canary_payload = {
        "weights": {"v1": 0.5, "v2": 0.5}
    }
    res = client.post("/v1/models/resnet/canary", json=canary_payload, headers=AUTH_HEADER)
    assert res.status_code == 200
    assert res.json()["canary_weights"]["v1"] == 0.5

    # Invalid weights not summing to 1.0
    bad_payload = {"weights": {"v1": 0.1, "v2": 0.2}}
    res_bad = client.post("/v1/models/resnet/canary", json=bad_payload, headers=AUTH_HEADER)
    assert res_bad.status_code == 400


def test_synchronous_predict_resnet():
    payload = {
        "input": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        "parameters": {"top_k": 3}
    }
    response = client.post("/v1/models/resnet/predict", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "resnet"
    assert "prediction" in data
    assert data["prediction"]["label"] is not None
    assert data["latency_ms"] >= 0
    assert "request_id" in data


def test_predict_pinned_version():
    payload = {"input": "test_input_tensor"}
    response = client.post("/v1/models/resnet/versions/v1/predict", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1"


def test_llm_text_generation():
    payload = {
        "prompt": "Dynamic batching in machine learning",
        "max_tokens": 50,
        "temperature": 0.7
    }
    response = client.post("/v1/models/llama/generate", json=payload, headers=AUTH_HEADER)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "llama"
    assert "generated_text" in data["prediction"]
    assert data["prediction"]["usage"]["total_tokens"] > 0


def test_async_job_lifecycle():
    payload = {
        "model": "resnet",
        "version": "v1",
        "input": "batch_job_image_sample",
        "priority": 5
    }
    # Create job
    create_resp = client.post("/v1/jobs", json=payload, headers=AUTH_HEADER)
    assert create_resp.status_code == 200
    job_id = create_resp.json()["job_id"]
    assert job_id.startswith("job_")
    assert create_resp.json()["status"] == "queued"

    # Query status
    status_resp = client.get(f"/v1/jobs/{job_id}", headers=AUTH_HEADER)
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id


def test_metrics_endpoint():
    res = client.get("/v1/metrics", headers=AUTH_HEADER)
    assert res.status_code == 200
    metrics = res.json()
    assert "requests_total" in metrics
    assert "p95_latency_ms" in metrics
    assert "active_workers" in metrics
