"""Offline wiring test.

Proves train -> save -> load -> score -> policy works end to end WITHOUT
downloading a real embedding model. A FakeEmbedder maps a few keywords to
separable vectors. This validates the plumbing, not semantics -- the real
SentenceTransformerEmbedder is what provides meaning on a networked machine.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "classifier"))

from classifier.embedder import Embedder


class FakeEmbedder(Embedder):
    """3-dim deterministic vectors: [hard-keyword count, easy-keyword count, length].
    Enough for logreg to separate -- stands in for real semantics in CI."""
    dim = 3
    model_name = "fake-embedder-for-tests"
    HARD = ["design", "prove", "analyze", "migration", "complexity", "trade-offs",
            "architect", "derive", "scalable", "throughput", "refactor", "plan"]
    EASY = ["capital", "translate", "define", "synonym", "color", "even",
            "backwards", "spell", "day comes"]

    def encode(self, texts):
        rows = []
        for t in texts:
            tl = t.lower()
            rows.append([
                sum(k in tl for k in self.HARD),
                sum(k in tl for k in self.EASY),
                len(t) / 100.0,
            ])
        return np.array(rows, dtype=float)


def main():
    fake = FakeEmbedder()

    import train
    train.main(embedder=fake)  # trains logreg on fake vectors, saves artifacts

    from router.signals import DifficultyClassifier
    from router.policy import RoutingPolicy
    clf = DifficultyClassifier(embedder=fake)   # loads saved logreg, uses same embedder
    policy = RoutingPolicy(clf)

    samples = [
        "What is the capital of Japan?",
        "Design a scalable zero-downtime migration and analyze the trade-offs.",
        "architect a high-throughput service",          # paraphrase of a hard prompt
        "import pandas; how do I read a CSV file?",
    ]
    print("\nend-to-end routing through the new embedding classifier:")
    for s in samples:
        d = policy.decide(s)
        print(f"  score={d.difficulty:.3f}  -> {d.tier:5s} {d.model:28s} | {d.reason}")


if __name__ == "__main__":
    main()
