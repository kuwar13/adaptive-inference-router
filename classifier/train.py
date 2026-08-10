"""Train the difficulty classifier: sentence-embeddings + logistic regression.

Only the logistic regression is trained -- the embedder is frozen/pretrained.
So training is: embed the labeled prompts once, fit logreg on the vectors, save.

Saves:
  difficulty_lr.joblib  -> the trained LogisticRegression
  classifier_meta.json  -> which embedder produced the vectors (so scoring uses
                           the SAME embedder) and its dimension
"""
from __future__ import annotations
import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import make_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
LR_OUT = os.path.join(HERE, "difficulty_lr.joblib")
META_OUT = os.path.join(HERE, "classifier_meta.json")


def main(embedder=None):
    texts, labels = make_dataset()
    y = np.array(labels)

    if embedder is None:
        from embedder import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

    X = embedder.encode(texts)                       # (N, dim) semantic vectors
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

    clf = LogisticRegression(max_iter=2000, C=2.0)   # the only thing we train
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    acc = accuracy_score(yte, (proba >= 0.5).astype(int))
    auc = roc_auc_score(yte, proba)

    joblib.dump(clf, LR_OUT)
    json.dump({"embedder": embedder.model_name, "dim": int(X.shape[1])}, open(META_OUT, "w"))
    print(f"trained (embeddings+logreg)  acc={acc:.3f}  auc={auc:.3f}  dim={X.shape[1]}")
    print(f"  -> {LR_OUT}")
    print(f"  -> {META_OUT} (embedder={embedder.model_name})")


if __name__ == "__main__":
    main()
