"""Model-layer tests: prediction shape and monitoring stats."""
from __future__ import annotations

from api.monitoring import Stats
from ml.model import predict_one


def test_predict_one_returns_label_and_probability():
    label, proba = predict_one(
        {
            "tenure_months": 10,
            "monthly_charges": 70.0,
            "total_charges": 700.0,
            "num_services": 3,
            "senior_citizen": 0,
            "contract_type": "one-year",
            "internet_service": "dsl",
            "tech_support": "yes",
            "payment_method": "credit-card",
        }
    )
    assert label in (0, 1)
    assert 0.0 <= proba <= 1.0


def test_stats_snapshot_math():
    s = Stats()
    s.record(churned=True, latency_ms=10.0)
    s.record(churned=False, latency_ms=20.0)
    snap = s.snapshot()
    assert snap["total_predictions"] == 2
    assert snap["predicted_churn"] == 1
    assert snap["churn_rate"] == 0.5
    assert snap["avg_latency_ms"] == 15.0
