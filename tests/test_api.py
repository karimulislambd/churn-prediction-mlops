"""API tests using FastAPI's TestClient. CI trains the model first (see ci.yml)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_high_risk_customer(client):
    """New, expensive, month-to-month fiber customer with no support -> should lean churn."""
    r = client.post(
        "/predict",
        json={
            "tenure_months": 1,
            "monthly_charges": 110.0,
            "total_charges": 110.0,
            "num_services": 2,
            "senior_citizen": 1,
            "contract_type": "month-to-month",
            "internet_service": "fiber",
            "tech_support": "no",
            "payment_method": "electronic-check",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk"] in {"low", "medium", "high"}


def test_predict_rejects_bad_input(client):
    r = client.post("/predict", json={"tenure_months": -5})
    assert r.status_code == 422  # pydantic validation error


def test_metrics_endpoint(client):
    client.post(
        "/predict",
        json={
            "tenure_months": 40,
            "monthly_charges": 30.0,
            "total_charges": 1200.0,
            "num_services": 5,
            "senior_citizen": 0,
            "contract_type": "two-year",
            "internet_service": "dsl",
            "tech_support": "yes",
            "payment_method": "bank-transfer",
        },
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "churn_predictions_total" in r.text
