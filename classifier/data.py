"""Build a labeled difficulty dataset.

PRODUCTION LABELING (what you say in an interview):
  Take historical prompts, run each through the small and the large model,
  score answer agreement (exact-match / embedding similarity / LLM judge).
  agreement high  -> label 0 (easy: small model was enough)
  agreement low   -> label 1 (hard: needed the large model)

PUBLIC-REPO LABELING (what runs here, offline, no GPU):
  We synthesize prompts from templates whose true difficulty we know, so the
  classifier has something real to learn. This is clearly a stand-in for the
  agreement-based labels above -- same pipeline, cheaper labels.
"""
from __future__ import annotations
import random

random.seed(7)

EASY_TEMPLATES = [
    "What is the capital of {x}?",
    "Translate '{x}' to French.",
    "Define the word {x}.",
    "What day comes after {x}?",
    "Convert {n} miles to kilometers.",
    "Summarize this in one line: {x} is a type of fruit.",
    "Is {n} an even number?",
    "Give me a synonym for {x}.",
    "What color is a ripe {x}?",
    "Spell the word {x} backwards.",
]
HARD_TEMPLATES = [
    "Prove that the sum of the first {n} odd numbers equals {n} squared, step by step.",
    "Design a rate limiter for a distributed API handling {n}k requests per second; discuss trade-offs.",
    "Given conflicting requirements {x} and {y}, derive an architecture and justify each decision.",
    "Refactor this recursive function to be iterative and analyze the new time complexity:\n\ndef f(n):\n    if n<2: return n\n    return f(n-1)+f(n-2)",
    "Explain why {x} fails under high concurrency and propose three fixes with pros and cons.",
    "Write a SQL query that finds the {n}th highest salary per department, then explain the plan.",
    "Analyze the ethical trade-offs of {x} versus {y} for a healthcare system.",
    "Derive the gradient of cross-entropy loss with respect to the logits and show each step.",
    "Plan a migration from {x} to {y} with zero downtime; list risks and rollbacks.",
    "Compare {x} and {y} across latency, cost, and consistency, and recommend one.",
]
# Moderate prompts: genuinely in-between. Labeled 50/50 at train time so the
# classifier is honestly uncertain about them -> probabilities land mid-band
# -> they route to the mid tier. This is what makes the mid model earn its keep.
MODERATE_TEMPLATES = [
    "Rewrite this paragraph about {x} to be more concise and professional.",
    "Draft a short product description for a {x} aimed at small businesses.",
    "Explain the difference between {x} and {y} to a beginner.",
    "Suggest {n} improvements for an onboarding email mentioning {x}.",
    "Turn these bullet points about {x} into a short paragraph.",
    "Write a polite follow-up message about a delayed {x} order.",
    "Outline a {n}-step plan to learn {x} in a month.",
    "Summarize the main pros and cons of using {x} for a side project.",
]

NOUNS = ["france", "japan", "brazil", "canada", "kenya", "banana", "apple", "mango",
         "kafka", "redis", "postgres", "kubernetes", "monday", "tuesday", "sunday",
         "python", "rust", "graphql", "rest", "sharding", "replication"]


def _fill(t: str) -> str:
    return t.format(x=random.choice(NOUNS), y=random.choice(NOUNS), n=random.randint(2, 500))


def make_dataset(n_per_class: int = 1500):
    rows = []
    for _ in range(n_per_class):
        rows.append((_fill(random.choice(EASY_TEMPLATES)), 0))
        rows.append((_fill(random.choice(HARD_TEMPLATES)), 1))
    # moderate prompts, deliberately split 50/50 so they sit mid-band
    for i in range(n_per_class):
        rows.append((_fill(random.choice(MODERATE_TEMPLATES)), i % 2))
    random.shuffle(rows)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    return texts, labels
