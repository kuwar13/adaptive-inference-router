"""Execution path: take a decision, run it with retry + fallback, record it.

The retry+fallback logic here is what earns the high completion rate: a request
that fails on its chosen model is retried once, then rerouted to a different
tier before it's allowed to count as a failure.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Optional

from .backends import ModelBackend, POOL
from .policy import RoutingPolicy, RoutingDecision


# Where to fall back to if a tier's pod is saturated/erroring.
FALLBACK = {"small": "mid", "code": "mid", "mid": "large", "large": "mid"}


@dataclass
class RequestRecord:
    decision: RoutingDecision
    success: bool
    attempts: int
    served_by: Optional[str]
    latency: float
    cost: float
    fell_back: bool


class Router:
    def __init__(self, policy: RoutingPolicy, backends: dict[str, ModelBackend]):
        self.policy = policy
        self.backends = backends  # keyed by tier: small/code/mid/large

    async def _attempt(self, tier: str, prompt_tokens: int, load_factor: float):
        return await self.backends[tier].generate(prompt_tokens, load_factor)

    async def handle(self, prompt: str, latency_slo: str = "interactive",
                     load_factor: float = 0.0, prompt_tokens: Optional[int] = None) -> RequestRecord:
        decision = self.policy.decide(prompt, latency_slo)
        if prompt_tokens is None:
            prompt_tokens = decision.context_tokens

        tier = decision.tier
        attempts = 0
        fell_back = False
        # try chosen tier (once), then retry once, then fall back to another tier once
        plan = [tier, tier, FALLBACK.get(tier, "large")]
        for i, t in enumerate(plan):
            attempts += 1
            try:
                res = await self._attempt(t, prompt_tokens, load_factor)
                if t != tier:
                    fell_back = True
                return RequestRecord(decision, True, attempts, res["model"],
                                     res["latency"], res["cost"], fell_back)
            except Exception:
                if i == len(plan) - 1:
                    return RequestRecord(decision, False, attempts, None, 0.0, 0.0, fell_back)
                await asyncio.sleep(0)  # yield; a real client backs off here
        # unreachable
        return RequestRecord(decision, False, attempts, None, 0.0, 0.0, fell_back)


def build_router(simulated: bool = True, **fail_kwargs) -> Router:
    """Factory used by the API, the benchmark and the load test."""
    from .signals import DifficultyClassifier
    clf = DifficultyClassifier()
    policy = RoutingPolicy(clf)
    if simulated:
        from .backends import SimulatedBackend
        backends = {t: SimulatedBackend(POOL[t], **fail_kwargs) for t in POOL}
    else:
        raise RuntimeError("Wire HuggingFaceBackend endpoints here for the cluster.")
    return Router(policy, backends)
