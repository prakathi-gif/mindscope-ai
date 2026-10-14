"""
Test suite for MindScope AI FastAPI inference service.
Validates input schemas, Pydantic validation, endpoints, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "MindScope AI"
    assert data["status"] == "Operational"
    assert "supported_classes" in data

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "loaded_models" in data
    assert "uptime_seconds" in data
    assert "memory_usage_mb" in data

def test_classes_metadata_endpoint():
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["num_classes"] == 8
    assert len(data["classes"]) == 8

def test_models_endpoint():
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_architectures" in data
    assert len(data["available_architectures"]) == 4

def test_predict_single_valid():
    payload = {
        "text": "I feel so hopeless every single day, completely hollow and drained of any joy or motivation to live.",
        "model": "logistic_regression",
        "return_probabilities": True,
        "redact_pii": True
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["predicted_id"] >= 0
    assert data["confidence"] > 0.0

def test_predict_single_invalid_empty():
    payload = {
        "text": "   ",
        "model": "logistic_regression"
    }
    response = client.post("/predict", json=payload)
    # Pydantic validation error should yield 422
    assert response.status_code == 422

def test_predict_batch():
    payload = {
        "texts": [
            "I cannot focus on anything today, constantly distracted, executive dysfunction.",
            "My chest is tight and pounding with pure panic and terrifying dread."
        ],
        "model": "logistic_regression",
        "redact_pii": True
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 2
    assert len(data["predictions"]) == 2

def test_explain_endpoint():
    payload = {
        "text": "I feel so depressed and hopeless and tired every day.",
        "top_k": 5
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_label" in data
    assert "top_attributions" in data
    assert len(data["top_attributions"]) <= 5

