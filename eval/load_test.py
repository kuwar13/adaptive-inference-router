"""Variable-workload load test -> the 'successful completion' number (bullet 2).

Generates bursty traffic (load_factor swings up and down), sends it through the
real routing + retry + fallback path against simulated backends whose failure
probability rises with load. A request only counts as failed if it fails on its
chosen tier, its retry, AND its fallback tier.
"""
from __future__ import annotations
import asyncio
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.engine import build_router
from classifier.data import EASY_TEMPLATES, MODERATE_TEMPLATES, HARD_TEMPLATES, _fill

random.seed(23)


def sample_prompt():
    r = random.random()
    if r < 0.22:
        return _fill(random.choice(EASY_TEMPLATES))
    if r < 0.37:
        return _fill(random.choice(MODERATE_TEMPLATES))
    return _fill(random.choice(HARD_TEMPLATES))


async def run(n=10000, waves=40):
    # Heavier failure model to represent real bursty saturation. Router retry +
    # fallback still recovers most of it; what's left is the completion gap.
    router = build_router(simulated=True, fail_base=0.02, fail_load_slope=0.24)
    results = []
    per = max(1, n // waves)
    for w in range(waves):
        # load swings between calm (~0.1) and heavy bursts (~1.0)
        load = 0.55 + 0.45 * math.sin(w / 2.0) + random.uniform(-0.1, 0.1)
        load = min(1.0, max(0.05, load))
        batch = [router.handle(sample_prompt(), load_factor=load) for _ in range(per)]
        results.extend(await asyncio.gather(*batch))

    total = len(results)
    ok = sum(r.success for r in results)
    fell_back = sum(r.fell_back for r in results)
    retried = sum(r.attempts > 1 for r in results)
    completion = ok / total

    print("=== variable-load test ===")
    print(f"requests            : {total}")
    print(f"completed           : {ok}")
    print(f"needed a retry/fb   : {retried} ({retried/total:.1%})")
    print(f"saved by fallback   : {fell_back} ({fell_back/total:.1%})")
    print(f"completion rate     : {completion:.3%}")

    try:
        from router.telemetry import log_summary
        log_summary("load_test",
                    {"requests": total, "waves": waves},
                    {"completion_rate": completion, "fallback_rate": fell_back / total,
                     "retry_rate": retried / total})
    except Exception as e:
        print(f"(mlflow logging skipped: {e})")


if __name__ == "__main__":
    asyncio.run(run())
