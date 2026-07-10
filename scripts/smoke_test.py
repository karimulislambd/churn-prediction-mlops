"""End-to-end smoke test: hit the running API in-process and print results.

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

HIGH_RISK = {
    "tenure_months": 1, "monthly_charges": 115.0, "total_charges": 115.0,
    "num_services": 2, "senior_citizen": 1, "contract_type": "month-to-month",
    "internet_service": "fiber", "tech_support": "no", "payment_method": "electronic-check",
}
LOW_RISK = {
    "tenure_months": 60, "monthly_charges": 25.0, "total_charges": 1500.0,
    "num_services": 6, "senior_citizen": 0, "contract_type": "two-year",
    "internet_service": "dsl", "tech_support": "yes", "payment_method": "bank-transfer",
}


def main() -> None:
    with TestClient(app) as c:
        print("[health]", c.get("/health").json())
        hi = c.post("/predict", json=HIGH_RISK).json()
        lo = c.post("/predict", json=LOW_RISK).json()
        print("[high-risk customer] ", hi)
        print("[low-risk customer]  ", lo)
        print("[dashboard stats]", c.get("/").json()["model"])

        ok = hi["churn_probability"] > lo["churn_probability"]
        print("\n" + ("PASS - model separates risk and API is live"
                      if ok else "CHECK - review output above"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
