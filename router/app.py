"""FastAPI service: the front desk.

POST /route  {prompt, latency_slo}  ->  routes, runs, logs, returns the answer
GET  /healthz                        ->  liveness/readiness for Kubernetes
"""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

from .engine import build_router
from . import telemetry

app = FastAPI(title="Adaptive Inference Router")
_router = None


class RouteRequest(BaseModel):
    prompt: str
    latency_slo: str = "interactive"  # or "batch"


def get_router():
    global _router
    if _router is None:
        _router = build_router(simulated=True)
    return _router


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/route")
async def route(req: RouteRequest):
    record = await get_router().handle(req.prompt, latency_slo=req.latency_slo)
    try:
        telemetry.log_request(record)
    except Exception:
        pass  # telemetry must never fail a request
    d = record.decision
    return {
        "model": record.served_by,
        "tier": d.tier,
        "reason": d.reason,
        "difficulty": round(d.difficulty, 3),
        "context_tokens": d.context_tokens,
        "success": record.success,
        "attempts": record.attempts,
        "fell_back": record.fell_back,
        "latency_estimate_s": round(record.latency, 3),
        "relative_cost": record.cost,
    }
