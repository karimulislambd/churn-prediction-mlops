"""Generate a realistic *synthetic* customer-churn dataset.

Synthetic (not real customer data) so the repo is self-contained and privacy-safe,
but with believable correlations: short tenure, month-to-month contracts, high
charges, and fiber-without-support all push churn probability up.

Run:  python -m data.generate_data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import CATEGORIES

RNG = np.random.default_rng(42)
N = 5000
OUT = Path(__file__).resolve().parent / "churn.csv"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate() -> pd.DataFrame:
    tenure = RNG.integers(0, 72, N)
    monthly = RNG.uniform(20, 120, N).round(2)
    num_services = RNG.integers(1, 8, N)
    senior = RNG.integers(0, 2, N)
    contract = RNG.choice(CATEGORIES["contract_type"], N, p=[0.55, 0.25, 0.20])
    internet = RNG.choice(CATEGORIES["internet_service"], N, p=[0.4, 0.45, 0.15])
    support = RNG.choice(CATEGORIES["tech_support"], N, p=[0.4, 0.6])
    payment = RNG.choice(CATEGORIES["payment_method"], N)

    total = (monthly * np.maximum(tenure, 1) * RNG.uniform(0.9, 1.1, N)).round(2)

    # Build a churn log-odds from believable drivers.
    logit = (
        -1.0
        + 0.035 * (monthly - 70)
        - 0.045 * tenure
        + 1.2 * (contract == "month-to-month")
        - 0.8 * (contract == "two-year")
        + 0.6 * (internet == "fiber")
        - 0.5 * (support == "yes")
        + 0.3 * senior
        + RNG.normal(0, 0.5, N)
    )
    churned = (RNG.uniform(0, 1, N) < _sigmoid(logit)).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_services": num_services,
            "senior_citizen": senior,
            "contract_type": contract,
            "internet_service": internet,
            "tech_support": support,
            "payment_method": payment,
            "churned": churned,
        }
    )


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows to {OUT}  (churn rate {df.churned.mean():.1%})")
