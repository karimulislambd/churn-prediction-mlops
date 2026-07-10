# Churn Prediction Service — end-to-end MLOps

[![CI](https://github.com/karimulislambd/churn-prediction-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/karimulislambd/churn-prediction-mlops/actions/workflows/ci.yml)
[![Live API](https://img.shields.io/badge/Live_API-docs-009688?logo=fastapi&logoColor=white)](https://churn-prediction-service.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-000000?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)

> A model **I train** (scikit-learn), served the production way: **FastAPI** with validated
> inputs and auto-generated docs, **Prometheus metrics**, a **live monitoring dashboard**,
> **Docker**, and **CI** that retrains and tests on every push.

Unlike a notebook or an API-wrapper project, this demonstrates the full lifecycle:
**data → train → version artifact → serve → monitor.**

**Live API docs (try it):** https://churn-prediction-service.onrender.com/docs
**Live monitoring dashboard:** https://churn-prediction-service.onrender.com/dashboard

> Hosted on Render's free tier, which sleeps after ~15 min idle — the first request
> may take ~30s to wake the service, and in-memory metrics reset on restart.

---

## What it does

Predicts whether a telecom customer will churn, from account attributes.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Churn probability + risk band for a customer (validated by Pydantic) |
| `GET /docs` | Interactive Swagger UI — **try the model in the browser** |
| `GET /dashboard` | Live monitoring: prediction count, churn rate, p95 latency (auto-refresh) |
| `GET /metrics` | Prometheus-format metrics for scraping |
| `GET /health` | Liveness + model-loaded check |

## Why this project

| Skill it demonstrates | Where |
|---|---|
| **Training pipeline** | Reproducible sklearn `Pipeline` + versioned artifact & metrics (`ml/train.py`) |
| **No train/serve skew** | One shared feature schema (`ml/features.py`), whole pipeline saved as one artifact |
| **Production serving** | FastAPI + Pydantic validation + typed responses (`api/`) |
| **Observability** | Prometheus metrics + a self-contained dashboard (`api/monitoring.py`) |
| **CI/CD** | GitHub Actions retrains the model and runs tests on every push |
| **Containerization** | Dockerfile pinned to Python 3.11; one-click Render deploy (`render.yaml`) |

## Architecture

```
 data/generate_data.py ─► data/churn.csv
                              │
                        ml/train.py ─► models/model.joblib + metrics.json
                              │
                     ┌────────┴─────────┐
                     ▼                  ▼
              FastAPI /predict    /dashboard + /metrics
             (Pydantic-validated)   (live monitoring)
```

## Model

GradientBoosting on ~5k synthetic-but-realistic customers. Held-out **ROC-AUC ≈ 0.81**.
The dataset is synthetic (privacy-safe, self-contained); the *engineering* is the point,
and the exact pipeline generalizes to any tabular problem.

## Quickstart

```bash
git clone https://github.com/karimulislambd/churn-prediction-mlops.git
cd churn-prediction-mlops

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python -m data.generate_data     # build the dataset
python -m ml.train               # train + save the artifact
uvicorn api.main:app --reload    # serve -> http://localhost:8000/docs
```

## Run the tests

```bash
python -m data.generate_data && python -m ml.train   # CI does this too
pytest -q
ruff check .
```

## Deploy (free)

Push to GitHub, then on [Render](https://render.com): **New → Blueprint → this repo**.
`render.yaml` + the Dockerfile do the rest; you get a public URL with live `/docs`.

## Roadmap

- [ ] Model registry + `/predict` A/B between versions
- [ ] Data-drift detection on incoming features
- [ ] Batch scoring endpoint (CSV in → CSV out)

---

Built by **Md Karimul Islam** — AI/ML Engineer · Computer Vision · LLM · XAI.
