"""The embedding step: sentence -> fixed-length semantic vector.

Replaces TF-IDF. The model is pre-trained and frozen -- we only download and
reuse it. Loaded once and kept in memory (constructing it per request would be
slow). encode() returns an (N, dim) numpy array; normalize_embeddings makes the
vectors unit-length, which plays nicely with a linear classifier.
"""
from __future__ import annotations
import numpy as np


class Embedder:
    """Interface. Anything with encode(list[str]) -> (N, dim) array works,
    which is what lets us inject a fake embedder in tests."""
    dim: int = 0
    model_name: str = "base"

    def encode(self, texts) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy: only needed on real runs
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()  # 384 for MiniLM

    def encode(self, texts) -> np.ndarray:
        return self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,          # batching keeps throughput high under load
        )
