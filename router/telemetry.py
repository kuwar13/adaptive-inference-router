"""MLflow telemetry.

Two granularities:
  - log_request(): per-request record (used by the FastAPI path; API rates are
    low enough that a run per request is fine).
  - log_summary(): aggregate metrics for a benchmark/load-test run.

Everything logs to a local ./mlruns store by default, so this works offline.
"""
from __future__ import annotations
import os
import mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))


def log_request(record) -> None:
    mlflow.set_experiment("routing-decisions")
    with mlflow.start_run(run_name="request"):
        d = record.decision
        mlflow.log_params({
            "chosen_tier": d.tier,
            "chosen_model": d.model,
            "reason": d.reason,
            "latency_slo": d.latency_slo,
            "verify": d.verify,
        })
        mlflow.log_metrics({
            "difficulty": d.difficulty,
            "context_tokens": d.context_tokens,
            "latency": record.latency,
            "cost": record.cost,
            "success": int(record.success),
            "attempts": record.attempts,
            "fell_back": int(record.fell_back),
        })


def log_summary(name: str, params: dict, metrics: dict) -> None:
    mlflow.set_experiment("routing-eval")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
