"""FastAPI churn-prediction service.

Endpoints:
  GET  /            -> service info
  GET  /health      -> liveness + whether the model is loaded
  POST /predict     -> churn prediction for a customer
  GET  /metrics     -> Prometheus metrics
  GET  /dashboard   -> human-readable live monitoring page
Interactive API docs are auto-generated at /docs (Swagger UI).
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.monitoring import LATENCY, PREDICTIONS, STATS
from api.schemas import Customer, Prediction
from ml.model import load_model, model_metrics, predict_one


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast at boot if the artifact is missing, and pay the load cost once.
    load_model()
    yield


app = FastAPI(
    title="Churn Prediction Service",
    description="Production-style ML inference API with monitoring. Try it at /docs.",
    version="1.0.0",
    lifespan=lifespan,
)


def _risk(p: float) -> str:
    return "high" if p >= 0.66 else "medium" if p >= 0.33 else "low"


@app.get("/")
def root() -> dict:
    return {
        "service": "churn-prediction",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "model": model_metrics(),
    }


@app.get("/health")
def health() -> dict:
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as exc:  # pragma: no cover - defensive
        return {"status": "degraded", "model_loaded": False, "error": str(exc)}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    start = time.perf_counter()
    label, proba = predict_one(customer.model_dump())
    latency = time.perf_counter() - start

    LATENCY.observe(latency)
    PREDICTIONS.labels(result="churn" if label else "retain").inc()
    STATS.record(bool(label), latency * 1000)

    return Prediction(
        churn=bool(label),
        churn_probability=round(proba, 4),
        risk=_risk(proba),
        model_version=model_metrics().get("model_version", "unknown"),
    )


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    s = STATS.snapshot()
    m = model_metrics()
    idle = s["total_predictions"] == 0
    hint = (
        '<div class="hint">No predictions yet — open '
        '<a href="/docs">/docs</a>, run <b>POST /predict</b> a few times, '
        "then refresh this page to see the metrics update live.</div>"
        if idle
        else ""
    )
    return f"""
    <html><head><title>Churn Service — Monitoring</title>
    <meta http-equiv="refresh" content="5">
    <style>
      *{{box-sizing:border-box}}
      body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;
            margin:0;padding:2.5rem 1.5rem;display:flex;justify-content:center}}
      .wrap{{width:100%;max-width:1100px}}
      h1{{margin:0 0 .25rem;font-size:1.6rem}}
      .sub{{color:#94a3b8;margin-bottom:1.75rem}}
      .live{{display:inline-block;width:9px;height:9px;border-radius:50%;background:#22c55e;
             margin-right:.4rem;animation:pulse 1.5s infinite}}
      @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
      .section{{color:#64748b;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;
                margin:1.75rem 0 .75rem}}
      .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem}}
      .card{{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:1.25rem}}
      .k{{color:#94a3b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em}}
      .v{{font-size:1.9rem;font-weight:700;margin-top:.35rem}}
      .hint{{background:#1e293b;border:1px dashed #475569;border-radius:14px;padding:1rem 1.25rem;
             margin-bottom:1.25rem;color:#cbd5e1}}
      a{{color:#60a5fa}}
      .foot{{margin-top:2rem;color:#64748b;font-size:.85rem}}
      .foot a{{margin-right:1.25rem}}
    </style></head><body><div class="wrap">
      <h1><span class="live"></span>Churn Prediction Service</h1>
      <div class="sub">Live monitoring · auto-refreshes every 5s · served from Render</div>
      {hint}
      <div class="section">Traffic</div>
      <div class="grid">
        <div class="card"><div class="k">Total predictions</div><div class="v">{s['total_predictions']}</div></div>
        <div class="card"><div class="k">Predicted churn</div><div class="v">{s['predicted_churn']}</div></div>
        <div class="card"><div class="k">Churn rate</div><div class="v">{s['churn_rate']*100:.1f}%</div></div>
        <div class="card"><div class="k">Avg latency</div><div class="v">{s['avg_latency_ms']} ms</div></div>
        <div class="card"><div class="k">p95 latency</div><div class="v">{s['p95_latency_ms']} ms</div></div>
        <div class="card"><div class="k">Uptime</div><div class="v">{s['uptime_seconds']}s</div></div>
      </div>
      <div class="section">Model</div>
      <div class="grid">
        <div class="card"><div class="k">Version</div><div class="v">v{m.get('model_version','?')}</div></div>
        <div class="card"><div class="k">Accuracy</div><div class="v">{m.get('accuracy','?')}</div></div>
        <div class="card"><div class="k">F1 score</div><div class="v">{m.get('f1','?')}</div></div>
        <div class="card"><div class="k">ROC-AUC</div><div class="v">{m.get('roc_auc','?')}</div></div>
        <div class="card"><div class="k">Train rows</div><div class="v">{m.get('n_train','?')}</div></div>
        <div class="card"><div class="k">Test rows</div><div class="v">{m.get('n_test','?')}</div></div>
      </div>
      <div class="foot">
        <a href="/docs">API docs</a>
        <a href="/metrics">Prometheus metrics</a>
        <a href="/health">Health</a>
      </div>
    </div></body></html>
    """
