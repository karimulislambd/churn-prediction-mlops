"""Lightweight observability: Prometheus metrics + in-memory stats for the dashboard.

Real production would scrape /metrics with Prometheus; the in-memory snapshot powers
a self-contained dashboard so the monitoring story is visible without extra infra.
"""
from __future__ import annotations

import threading
import time

from prometheus_client import Counter, Histogram

PREDICTIONS = Counter("churn_predictions_total", "Total predictions", ["result"])
LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency (s)")


class Stats:
    """Thread-safe rolling snapshot for the human-readable dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0
        self.churn = 0
        self.latencies_ms: list[float] = []
        self.started = time.time()

    def record(self, churned: bool, latency_ms: float) -> None:
        with self._lock:
            self.total += 1
            if churned:
                self.churn += 1
            self.latencies_ms.append(latency_ms)
            if len(self.latencies_ms) > 1000:
                self.latencies_ms.pop(0)

    def snapshot(self) -> dict:
        with self._lock:
            lat = self.latencies_ms
            avg = sum(lat) / len(lat) if lat else 0.0
            p95 = sorted(lat)[int(len(lat) * 0.95) - 1] if lat else 0.0
            return {
                "total_predictions": self.total,
                "predicted_churn": self.churn,
                "churn_rate": round(self.churn / self.total, 3) if self.total else 0.0,
                "avg_latency_ms": round(avg, 2),
                "p95_latency_ms": round(p95, 2),
                "uptime_seconds": int(time.time() - self.started),
            }


STATS = Stats()
