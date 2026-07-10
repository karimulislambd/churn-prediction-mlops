"""Train the churn model and save a versioned artifact + metrics.

The whole preprocessing + model is saved as ONE sklearn Pipeline, so serving is a
single `joblib.load` with no feature-engineering code duplicated at inference time.

Run:  python -m ml.train
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "churn.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
MODEL_VERSION = "1.0.0"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Saved model ->", MODEL_PATH)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
