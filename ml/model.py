"""Model loading + prediction wrapper used by the API."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run `python -m ml.train` first."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def model_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def predict_one(features: dict) -> tuple[int, float]:
    """Return (label, churn_probability) for a single customer record."""
    row = pd.DataFrame([{k: features[k] for k in FEATURES}])
    model = load_model()
    proba = float(model.predict_proba(row)[0, 1])
    return int(proba >= 0.5), proba
