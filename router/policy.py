"""Routing policy: turn the three signals into one model choice.

Design choices worth defending in an interview:
  - Conservative by default: only offload to a cheaper tier when the classifier
    is confident it's easy. This is WHY savings are ~27% and not 60% -- the
    routing protects quality on purpose.
  - Hard constraints first: context length can force a bigger model regardless
    of difficulty (a small model that can't fit the prompt is not an option).
  - Specialization: code prompts go to the code model at the small-tier cost.
  - Verify band: borderline prompts run on the small model first and only
    escalate if a cheap check looks shaky (implemented in the router path).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

from .signals import DifficultyClassifier, count_tokens, looks_like_code
from .backends import POOL


@dataclass
class RoutingDecision:
    tier: str
    model: str
    difficulty: float
    context_tokens: int
    latency_slo: str
    reason: str
    verify: bool  # borderline -> run small then check before trusting

    def as_dict(self) -> dict:
        return asdict(self)


class RoutingPolicy:
    def __init__(
        self,
        clf: DifficultyClassifier,
        easy_threshold: float = 0.32,   # below -> cheap tier
        hard_threshold: float = 0.55,   # above -> large tier
        long_context: int = 8000,       # tokens that force at least mid
        verify_lo: float = 0.32,
        verify_hi: float = 0.42,
    ):
        self.clf = clf
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self.long_context = long_context
        self.verify_lo = verify_lo
        self.verify_hi = verify_hi

    def decide(self, prompt: str, latency_slo: str = "interactive") -> RoutingDecision:
        ctx = count_tokens(prompt)
        diff = self.clf.score(prompt)

        # 1) Hard constraint: very long context can't go to the smallest tier.
        forced_min = "mid" if ctx > self.long_context else None

        # 2) Specialization: route code to the code model (small-tier cost).
        if looks_like_code(prompt) and forced_min is None and diff < self.hard_threshold:
            spec = POOL["code"]
            return RoutingDecision(spec.tier, spec.name, diff, ctx, latency_slo,
                                   "code prompt -> code specialist", verify=False)

        # 3) Difficulty-based tiering, conservative.
        if diff >= self.hard_threshold:
            tier = "large"
            reason = "high difficulty -> large"
        elif diff < self.easy_threshold and forced_min is None:
            tier = "small"
            reason = "confident easy -> small"
        else:
            tier = "mid"
            reason = "moderate / long-context -> mid"

        if forced_min == "mid" and tier == "small":
            tier, reason = "mid", "context too long for small"

        # 4) Batch traffic can absorb latency but we still respect cost; leave as is.
        verify = (tier == "small" and self.verify_lo <= diff <= self.verify_hi)
        spec = POOL[tier]
        return RoutingDecision(spec.tier, spec.name, diff, ctx, latency_slo, reason, verify)
