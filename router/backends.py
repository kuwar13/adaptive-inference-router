"""Model pool + pluggable backends.

Two implementations of the same interface:
  - SimulatedBackend: runs anywhere, no GPU/network. Latency and failure
    probability both grow with load, so load tests are meaningful.
  - HuggingFaceBackend: the real thing (TGI / transformers). Code is here
    for reference; it only runs where a GPU + HF access exist.

Costs are RELATIVE GPU-seconds per request, roughly tracking model size.
The all-large baseline uses cost_per_request of the large tier for every
request; routing beats that by sending easy/short traffic to cheaper tiers.
"""
from __future__ import annotations
import asyncio
import random
from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    tier: str            # "small" | "mid" | "large" | "code"
    cost_per_request: float
    base_latency: float  # seconds at zero load
    max_context: int     # token limit of the model


# The pool. One model family (Qwen2.5) so quality scales predictably across
# tiers, plus a code-specialized model so "specialized" is literally true.
POOL = {
    "small": ModelSpec("Qwen2.5-7B-Instruct",       "small", cost_per_request=1.0, base_latency=0.25, max_context=32000),
    "code":  ModelSpec("Qwen2.5-Coder-7B-Instruct", "code",  cost_per_request=1.0, base_latency=0.28, max_context=32000),
    "mid":   ModelSpec("Qwen2.5-32B-Instruct",      "mid",   cost_per_request=3.5, base_latency=0.55, max_context=32000),
    "large": ModelSpec("Qwen2.5-72B-Instruct",      "large", cost_per_request=8.0, base_latency=1.10, max_context=32000),
}


class ModelBackend:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    async def generate(self, prompt_tokens: int, load_factor: float = 0.0) -> dict:
        raise NotImplementedError


class SimulatedBackend(ModelBackend):
    """Deterministic-enough stand-in used for CI, cost benchmarks and load tests.

    fail_base       : baseline per-attempt failure probability
    fail_load_slope : how sharply failure rises as the pod saturates
    """
    def __init__(self, spec: ModelSpec, fail_base: float = 0.004, fail_load_slope: float = 0.05):
        super().__init__(spec)
        self.fail_base = fail_base
        self.fail_load_slope = fail_load_slope

    async def generate(self, prompt_tokens: int, load_factor: float = 0.0) -> dict:
        # Reported latency reflects size + tokens + load; real sleep is capped
        # so a 10k-request load test finishes in seconds.
        reported_latency = self.spec.base_latency * (1 + 0.6 * load_factor) + prompt_tokens * 0.00015
        await asyncio.sleep(min(reported_latency, 0.01))
        fail_p = self.fail_base + self.fail_load_slope * load_factor
        if random.random() < fail_p:
            raise RuntimeError(f"{self.spec.name} timed out / errored under load")
        return {
            "model": self.spec.name,
            "tier": self.spec.tier,
            "latency": reported_latency,
            "cost": self.spec.cost_per_request,
            "prompt_tokens": prompt_tokens,
        }


class HuggingFaceBackend(ModelBackend):
    """Reference implementation for a real cluster. Not exercised in the sandbox.

    Points at a Text Generation Inference (TGI) endpoint served per model as a
    Kubernetes Deployment. Swap this in by constructing the router with
    HuggingFaceBackend(spec, endpoint=...) instead of SimulatedBackend.
    """
    def __init__(self, spec: ModelSpec, endpoint: str, timeout: float = 8.0):
        super().__init__(spec)
        self.endpoint = endpoint
        self.timeout = timeout

    async def generate(self, prompt: str, load_factor: float = 0.0) -> dict:  # type: ignore[override]
        import time
        import httpx  # imported lazily; only needed on the cluster
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.endpoint}/generate",
                json={"inputs": prompt, "parameters": {"max_new_tokens": 512}},
            )
            r.raise_for_status()
            data = r.json()
        return {
            "model": self.spec.name,
            "tier": self.spec.tier,
            "latency": time.perf_counter() - t0,
            "cost": self.spec.cost_per_request,
            "text": data.get("generated_text", ""),
        }
