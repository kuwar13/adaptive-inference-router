# Adaptive Inference Routing System

A request-routing layer that sends each prompt to the **cheapest model that can still
answer it well**. Every request is scored on three signals — difficulty, context length,
and latency requirement — then routed across small / mid / large model APIs. A retry +
fallback path keeps completion high under load, and every routing decision is logged to
MLflow.

**Stack:** Python · FastAPI · OpenAI API · sentence-transformers · scikit-learn · Kubernetes · MLflow

## Results

| Goal | How it's measured | Result |
|---|---|---|
| Lower average inference cost | `eval/cost_benchmark.py` — routed cost vs an all-large baseline | **~27% reduction** |
| High completion under variable load | `eval/load_test.py` — bursty traffic through retry + fallback | **~99.3% completion** |

Both numbers emerge from the code on a stated workload / failure model — nothing is hardcoded.

## How it works

1. **Signals** (`router/signals.py`) — context length (token count) and latency (an SLO tag)
   are plain code; difficulty comes from a small classifier: a frozen sentence-embedding
   model (`all-MiniLM-L6-v2`) → logistic regression → probability the prompt is hard.
2. **Policy** (`router/policy.py`) — combines the signals into a tier. Conservative: only
   offloads to a cheaper model when the classifier is confident. Long context forces a bigger tier.
3. **Engine** (`router/engine.py`) — calls the chosen model, retries once, then falls back to
   another tier. A request only fails if all three attempts fail.
4. **Telemetry** (`router/telemetry.py`) — logs every routing decision, latency, and cost to MLflow.

The router runs as a stateless FastAPI service on Kubernetes (autoscaled via HPA); MLflow
runs alongside it. The models are hosted APIs, so there are no GPU pods to manage.

## Model pool

Same model family, three sizes, so quality scales predictably. Cost weights are the real
per-token price ratios (Luna = 1×).

| Tier | Model | Rel. cost | Used for |
|---|---|---|---|
| small | GPT-5.6 Luna | 1 | confident-easy prompts |
| mid | GPT-5.6 Terra | 10 | moderate / long-context / code prompts |
| large | GPT-5.6 Sol | 25 | hard prompts (this is the baseline) |

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python tests/test_wiring.py     # offline: proves the pipeline with a fake embedder
python classifier/train.py      # downloads the embedder, fits logreg, saves artifacts
python eval/cost_benchmark.py   # prints the cost-reduction number
python eval/load_test.py        # prints the completion-rate number
uvicorn router.app:app --reload # serve the API, then POST /route
```

## Honest limitations

- The 27% / 99.3% depend on the workload and failure model in `eval/`.
- Model backends are **simulated** in this repo; the API backend is the production path.
- Difficulty labels are synthetic here; production uses small-vs-large answer agreement on
  real traffic — the one step between prototype and production.
