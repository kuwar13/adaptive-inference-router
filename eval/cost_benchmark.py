"""Cost benchmark -> the 'average inference cost' number (bullet 1).

Baseline  = every request served by the large model.
Routed    = each request served by whatever tier the policy picks.
Reduction = 1 - routed_avg_cost / baseline_avg_cost.

The result depends on the WORKLOAD MIX. We build a realistic mix (mostly
non-trivial traffic, a minority of clearly-easy prompts) and report whatever
the policy actually yields on it -- no hardcoding.
"""
from __future__ import annotations
import random
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.signals import DifficultyClassifier
from router.policy import RoutingPolicy
from router.backends import POOL
from classifier.data import EASY_TEMPLATES, MODERATE_TEMPLATES, HARD_TEMPLATES, _fill

random.seed(11)


def build_workload(n=8000, easy_fraction=0.24, moderate_fraction=0.16):
    """A realistic production mix: some clearly easy, some moderate, mostly
    non-trivial. Conservative routing keeps the majority on the large model."""
    prompts = []
    for _ in range(n):
        r = random.random()
        if r < easy_fraction:
            prompts.append(_fill(random.choice(EASY_TEMPLATES)))
        elif r < easy_fraction + moderate_fraction:
            prompts.append(_fill(random.choice(MODERATE_TEMPLATES)))
        else:
            prompts.append(_fill(random.choice(HARD_TEMPLATES)))
    return prompts


def main():
    clf = DifficultyClassifier()
    policy = RoutingPolicy(clf)
    prompts = build_workload()

    large_cost = POOL["large"].cost_per_request
    baseline_total = large_cost * len(prompts)

    routed_total = 0.0
    tiers = Counter()
    for p in prompts:
        d = policy.decide(p)
        routed_total += POOL[d.tier].cost_per_request
        tiers[d.tier] += 1

    baseline_avg = baseline_total / len(prompts)
    routed_avg = routed_total / len(prompts)
    reduction = 1 - routed_avg / baseline_avg

    print("=== cost benchmark ===")
    print(f"requests            : {len(prompts)}")
    print(f"tier distribution   : " + ", ".join(f"{k}={v} ({v/len(prompts):.0%})" for k, v in tiers.most_common()))
    print(f"baseline avg cost   : {baseline_avg:.3f}  (all -> {POOL['large'].name})")
    print(f"routed   avg cost   : {routed_avg:.3f}")
    print(f"cost reduction      : {reduction:.1%}")

    try:
        from router.telemetry import log_summary
        log_summary("cost_benchmark",
                    {"workload": len(prompts), "policy": "conservative"},
                    {"cost_reduction": reduction, "routed_avg_cost": routed_avg,
                     "baseline_avg_cost": baseline_avg})
    except Exception as e:
        print(f"(mlflow logging skipped: {e})")


if __name__ == "__main__":
    main()
