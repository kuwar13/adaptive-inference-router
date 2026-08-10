"""The three routing signals.

Only difficulty needs a model. It is now:
  embedding model (frozen, semantic) -> logistic regression -> probability.
Same score() interface as the old TF-IDF version, so the policy/engine/app are
unchanged -- the featurizer swap is fully contained here.

Length and urgency are still pure code (see count_tokens / the SLO tag).
"""
from __future__ import annotations
import os
import re
import json
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
LR_PATH = os.path.join(HERE, "..", "classifier", "difficulty_lr.joblib")
META_PATH = os.path.join(HERE, "..", "classifier", "classifier_meta.json")

_CODE_HINT = re.compile(r"```|def |class |import |SELECT |function |=>|{\s*\n|;\s*$", re.MULTILINE)


def count_tokens(text: str) -> int:
    """Cheap word/punct approximation of a tokenizer -- fine for a routing
    threshold. A real deploy uses the model's own tokenizer."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def looks_like_code(text: str) -> bool:
    return bool(_CODE_HINT.search(text))


class DifficultyClassifier:
    """embedding -> logistic regression -> P(hard) in [0, 1].

    Pass an embedder to inject a fake one in tests; otherwise it loads the same
    SentenceTransformer that produced the training vectors (from meta)."""
    def __init__(self, embedder=None, lr_path: str = LR_PATH, meta_path: str = META_PATH):
        self.clf = joblib.load(lr_path)
        if embedder is None:
            from classifier.embedder import SentenceTransformerEmbedder
            meta = json.load(open(meta_path))
            embedder = SentenceTransformerEmbedder(meta["embedder"])
        self.embedder = embedder

    def score(self, prompt: str) -> float:
        vec = self.embedder.encode([prompt])            # sentence -> (1, dim)
        return float(self.clf.predict_proba(vec)[0][1]) # (1, dim) -> P(hard)
