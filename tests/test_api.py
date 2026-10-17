
"""API integration tests for Kestrel routing."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_endpoint():
    response = client.post("/predict", json={
        "request_id": "TEST-DEMO-001",
        "channel": "whatsapp",
        "product_family": "Water Purifier",
        "warranty_status": "in_warranty",
        "request_text": "My water purifier has stopped working and needs repair.",
        "source": "crm",
    })

    assert response.status_code == 200

    result = response.json()

    assert result["request_id"] == "TEST-DEMO-001"
    assert result["predicted_team"] in {
        "Repairs",
        "Billing",
        "Product Advice",
        "Returns & Replacement",
        "Warranty Claims",
        "Installs & Demo",
        "Filters & Consumables",
    }
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["reasons"], list)
    assert isinstance(result["review_required"], bool)


def test_invalid_request_is_rejected():
    response = client.post("/predict", json={
        "channel": "whatsapp",
        "product_family": "Water Purifier",
    })

    assert response.status_code == 422
