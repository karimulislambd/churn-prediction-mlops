"""Feature schema shared by training and serving.

Keeping the column definitions in one place guarantees the API validates exactly
the features the model was trained on — a common source of train/serve skew.
"""
from __future__ import annotations

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_services",
    "senior_citizen",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "tech_support",
    "payment_method",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churned"

# Allowed values for categorical fields (used for validation + data generation).
CATEGORIES = {
    "contract_type": ["month-to-month", "one-year", "two-year"],
    "internet_service": ["dsl", "fiber", "none"],
    "tech_support": ["yes", "no"],
    "payment_method": ["electronic-check", "mailed-check", "bank-transfer", "credit-card"],
}
