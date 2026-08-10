# Adaptive Inference Routing System

A request-routing layer that picks the cheapest language model that can still
answer well. Every request is scored on three signals before any model runs;
easy/short traffic goes to small models, hard/long traffic to large ones, with
retry + fallback for reliability. All routing decisions, latencies and quality
signals are logged to MLflow.

**Stack:** Python, FastAPI, Hugging Face (TGI), Kubernetes, MLflow, sentence-transformers, scikit-learn

---

## The two results this repo reproduces

| Claim | How it's produced | Measured here |
|---|---|---|
| Lower average inference cost | `eval/cost_benchmark.py` compares routed cost vs an all-large baseline over a realistic workload | **~27% reduction** |
| High successful completion under variable load | `eval/load_test.py` drives bursty traffic through retry + fallback against backends that fail more as they saturate | **~99.3% completion** |

Both numbers *emerge* from the code on a stated workload/failure scenario — nothing is hardcoded. Change the workload mix or failure model and the numbers move; the assumptions are all in those two files.

---

## Architecture

```
request ──► FastAPI router ──► [ signals ] ──► [ policy ] ──► [ engine ] ──► model pool
                                   │              │              │
                       length (token count)   pick tier    retry + fallback
                       urgency (SLO tag)       (conservative) │
                       difficulty (classifier)                └──► MLflow telemetry
```

- **signals.py** — the three routing signals. Only difficulty needs a model
  (a tiny TF-IDF + logistic-regression classifier); length and urgency are
  plain code.
- **policy.py** — combines signals into a tier choice. Conservative by design:
  only offloads to a cheaper tier when the classifier is confident. Hard
  constraints (context length) come first; code prompts go to a code model.
- **engine.py** — runs the choice with one retry then a fallback tier. This is
  what earns the completion rate.
- **backends.py** — `SimulatedBackend` (runs anywhere) and `HuggingFaceBackend`
  (real TGI endpoints on the cluster). Same interface, swap at construction.
- **telemetry.py** — MLflow logging, per-request and per-eval-run.
- **k8s/** — router + per-model Deployments, Services and HPAs.

## The model pool

One family so quality scales predictably across sizes, plus a code specialist so
"specialized models" is literally true:

| Tier | Model | Rel. cost | Used for |
|---|---|---|---|
| small | Qwen2.5-7B-Instruct | 1.0 | confident-easy prompts |
| code | Qwen2.5-Coder-7B-Instruct | 1.0 | code prompts |
| mid | Qwen2.5-32B-Instruct | 3.5 | moderate / long context |
| large | Qwen2.5-72B-Instruct | 8.0 | hard prompts (the baseline) |

## How the difficulty classifier works

`embedding model (frozen) -> logistic regression -> P(hard)`. A pre-trained
sentence-embedding model (`all-MiniLM-L6-v2`) turns each prompt into a 384-dim
semantic vector; a logistic regression drawn over those vectors outputs the
difficulty probability. Only the logistic regression is trained -- the embedder
is downloaded and frozen -- so training is "embed the labeled prompts once, fit
logreg, save." Because it keys on *meaning*, it generalizes to prompts phrased
in vocabulary it never saw (a paraphrase of a hard prompt still scores high),
which a bag-of-words model cannot do.

The featurizer swap is fully contained in `DifficultyClassifier` and
`classifier/embedder.py`; `policy.py`, `engine.py` and `app.py` are unchanged.
`tests/test_wiring.py` proves the whole path offline with an injected fake
embedder (keyword-based, so it validates plumbing, not semantics).

## How the labels are made

In production: run historical prompts through the small and large models, score
answer agreement (embedding similarity / LLM judge). High agreement -> easy
(label 0); low agreement -> hard (label 1). In this repo (offline): labels come
from templates of known difficulty -- a documented stand-in for those labels.

---

## Run it

```bash
pip install -r requirements.txt
python tests/test_wiring.py         # offline: proves the pipeline with a fake embedder
python classifier/train.py          # downloads the embedder, fits logreg, saves artifacts
python eval/cost_benchmark.py       # prints the cost-reduction number
python eval/load_test.py            # prints the completion-rate number
uvicorn router.app:app --reload     # serve the API, then POST /route
mlflow ui                           # view logged runs
```

`train.py` needs network access the first time (to fetch the embedding model).
`test_wiring.py` needs nothing and runs anywhere.

Example request:

```bash
curl -s localhost:8000/route -H 'content-type: application/json' \
  -d '{"prompt":"What is the capital of Japan?","latency_slo":"interactive"}'
```

## Notes / honest limitations

- The 27% and 99.3% depend on the workload and failure assumptions in `eval/`;
  they're a stated scenario, not a universal constant.
- Simulated backends stand in for real model serving so the whole thing runs
  without GPUs; the `HuggingFaceBackend` path is the production one.
- Synthetic difficulty labels are separable enough that the classifier scores
  high; real agreement labels are noisier (that's expected and fine).
