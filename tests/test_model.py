
"""Tests for the trained routing model."""

from app.predictor import RoutingPredictor
from src.data import VALID_TEAMS


def test_model_loads_and_predicts():
    predictor = RoutingPredictor()

    result = predictor.predict({
        "channel": "whatsapp",
        "product_family": "Water Purifier",
        "warranty_status": "in_warranty",
        "request_text": "My water purifier has stopped working and needs repair.",
        "source": "crm",
    })

    assert result["predicted_team"] in VALID_TEAMS
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_name"]
    assert result["reasons"]


def test_prediction_probabilities_are_valid():
    predictor = RoutingPredictor()

    result = predictor.predict({
        "channel": "email",
        "product_family": "Air Fryer",
        "warranty_status": "out_of_warranty",
        "request_text": "I need help with my appliance.",
        "source": "crm",
    })

    probabilities = result["probabilities"]

    assert set(probabilities) == VALID_TEAMS
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
