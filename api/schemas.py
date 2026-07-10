"""Pydantic request/response models — this is the API's validation contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Customer(BaseModel):
    """One customer record. Field constraints reject bad input before it hits the model."""

    tenure_months: int = Field(..., ge=0, le=120, examples=[5])
    monthly_charges: float = Field(..., ge=0, le=1000, examples=[89.5])
    total_charges: float = Field(..., ge=0, examples=[450.0])
    num_services: int = Field(..., ge=0, le=20, examples=[3])
    senior_citizen: int = Field(..., ge=0, le=1, examples=[0])
    contract_type: Literal["month-to-month", "one-year", "two-year"] = "month-to-month"
    internet_service: Literal["dsl", "fiber", "none"] = "fiber"
    tech_support: Literal["yes", "no"] = "no"
    payment_method: Literal[
        "electronic-check", "mailed-check", "bank-transfer", "credit-card"
    ] = "electronic-check"


class Prediction(BaseModel):
    churn: bool
    churn_probability: float
    risk: Literal["low", "medium", "high"]
    model_version: str
