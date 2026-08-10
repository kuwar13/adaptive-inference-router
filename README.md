# Adaptive Inference Routing System

A request-routing layer that sends each prompt to the **cheapest model that can still
answer it well**. Every request is scored on three signals before any model runs —
prompt difficulty, context length, and latency requirement — then routed across a pool
of models of different sizes. A retry + fallback path keeps completion high under load,
and every routing decision is logged to MLflow.

**Stack:** Python · FastAPI · sentence-transformers · scikit-learn · Hugging Face TGI · Kubernetes · MLflow

---

## Results

| Goal | How it's measured | Result |
|---|---|---|
| Lower average inference cost | `eval/cost_benchmark.py` — routed cost vs an all-large baseline over a realistic workload | **~27% reduction** |
| High completion under variable load | `eval/load_test.py` — bursty traffic through retry + fallback against load-scaling failures | **~99.3% completion** |

Both numbers *emerge from the code* on a stated workload and failure model — nothing is
hardcoded. Change the assumptions in `eval/` and the numbers move.

---

## Architecture

```
request
   │
   ▼
FastAPI router ──▶ signals ──▶ policy ──▶ engine ──▶ model pool
                     │           │          │
        length (token count)  pick tier   retry + fallback
        latency (SLO tag)     (conservative)  │
        difficulty (classifier)               └──▶ MLflow telemetry
```

- **`router/signals.py`** — the three routing signals. Only difficulty needs a model
  (embedding + logistic regression); length and urgency are plain code.
- **`router/policy.py`** — combines the signals into a tier choice. Conservative by
  design: only offloads to a cheaper tier when the classifier is confident. Context
  length is a hard constraint; code prompts route to a code specialist.
- **`router/engine.py`** — runs the chosen tier with one retry, then a fallback tier.
  This is what earns the completion rate.
- **`router/backends.py`** — `SimulatedBackend` (runs anywhere, used for dev/tests/evals)
  and `HuggingFaceBackend` (real TGI endpoints on the cluster). Same interface, swap at
  construction — so the routing logic is backend-agnostic.
- **`router/telemetry.py`** — MLflow logging, per-request and per-eval-run.
- **`k8s/`** — router + per-model Deployments, Services, and HPAs.

## Model pool

One model family so quality scales predictably across sizes, plus a code specialist so
"specialized models" is literal. Costs are relative GPU-seconds per request.

| Tier | Model | Rel. cost | Used for |
|---|---|---|---|
| small | Qwen2.5-7B-Instruct | 1.0 | confident-easy prompts |
| code | Qwen2.5-Coder-7B-Instruct | 1.0 | code prompts (below the hard-difficulty threshold) |
| mid | Qwen2.5-32B-Instruct | 3.5 | moderate / long-context prompts |
| large | Qwen2.5-72B-Instruct | 8.0 | hard prompts (this is the baseline) |

## How the difficulty classifier works

`embedding model (frozen) → logistic regression → P(hard)`. A pretrained
sentence-embedding model (`all-MiniLM-L6-v2`) turns each prompt into a 384-dim semantic
vector; a logistic regression over those vectors outputs the difficulty probability.
Only the logistic regression is trained — the embedder is downloaded and frozen — so
training is "embed the labeled prompts once, fit logreg, save." Because it keys on
*meaning*, it generalizes to prompts phrased in vocabulary it never saw (a paraphrase of
a hard prompt still scores high), which a bag-of-words model can't do.

The featurizer is fully isolated behind `DifficultyClassifier.score()`, so `policy.py`,
`engine.py`, and `app.py` never change when it's swapped.

## How the labels are made

**Production:** run historical prompts through the small and large models, score answer
agreement (embedding similarity / LLM judge). High agreement → easy (label 0); low
agreement → hard (label 1).

**This repo (offline):** labels are synthesized from templates of known difficulty — a
documented stand-in for the agreement labels, same training pipeline.

---

## Run it

```bash
pip install -r requirements.txt

python tests/test_wiring.py     # offline: proves the full pipeline with a fake embedder
python classifier/train.py      # downloads the embedder, fits logreg, saves artifacts
python eval/cost_benchmark.py   # prints the cost-reduction number
python eval/load_test.py        # prints the completion-rate number
uvicorn router.app:app --reload # serve the API, then POST /route
mlflow ui                       # view logged routing decisions and eval runs
```

Example request:

```bash
curl -s localhost:8000/route -H 'content-type: application/json' \
  -d '{"prompt":"What is the capital of Japan?","latency_slo":"interactive"}'
```

`train.py` needs network access the first time (to fetch the embedding model).
`test_wiring.py` needs nothing and runs anywhere.

---

## Design decisions

- **Cheap router.** The routing decision is two regex checks plus one small classifier —
  sub-millisecond. It has to cost far less than the model call it's trying to avoid.
- **Conservative routing.** Savings are ~27%, not 60%, on purpose: only confidently-easy
  traffic is offloaded, so answer quality is protected. The number is a cost-vs-quality
  knob, not a ceiling.
- **Retry before fallback.** Most failures under load are transient, so a same-tier retry
  clears them; falling back to another tier is the heavier move.
- **Pluggable backends.** Simulated backends make the routing logic deterministically
  load-testable without GPUs; the TGI backend is the production path.

## Honest limitations

- The 27% and 99.3% depend on the workload mix and failure model stated in `eval/`.
- Model backends are **simulated** in this repo (no GPUs required); the `HuggingFaceBackend`
  path is what runs on a real cluster.
- Difficulty labels are synthetic here; production uses small-vs-large answer agreement on
  real traffic. That label source is the one step between prototype and production.
- The load test validates routing/fallback **logic**, not wall-clock throughput; a real
  throughput test would drive the live service with k6 or Locust.
