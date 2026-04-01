# PKPD Agent

Agentic PK modelling pipeline based on the Acocella 1984 study.
Fits a one-compartment IV infusion model on raw concentration-time data using an iterative agent loop, with optional LLM-powered paper extraction.

Deployable as a REST API on **Google Cloud Run**, with **Gemini** as the default LLM provider.

---

## How it works

The agent loop runs up to 5 iterations. At each iteration:

1. `agent_inspect` — flags outlier subjects (RMSE > 2× median) for exclusion
2. `agent_fit_individual` — grid search per subject, within refined CL/V bounds
3. `agent_fit_pooled` — pooled grid search, then zooms the search grid for next iteration
4. `agent_read_paper` — extracts model details from the reference PDF via LLM (once)
5. `agent_report` — writes results CSV and markdown report

Each iteration narrows the parameter search space and excludes outliers detected in the previous run — the loop converges on progressively better fits.

---

## Quickstart (CLI)

```bash
uv venv && uv pip install -e ".[gemini]"
export GOOGLE_API_KEY="your_key"
python3 -m poc.agent_poc
```

Outputs:
- `poc/results.csv` — per-subject CL, V, RMSE
- `poc/report.md` — full run report with paper insights

---

## API (Cloud Run)

### Run locally

```bash
uv pip install -e ".[gemini]"
export GOOGLE_API_KEY="your_key"
uvicorn poc.app:app --reload
```

| Port | Usage |
|------|-------|
| `8000` | `uvicorn --reload` (dev local) |
| `8080` | Docker / Cloud Run |

### Run via Docker

```bash
docker build -t pkpd-agent .
docker run -p 8080:8080 -e GOOGLE_API_KEY=your_key pkpd-agent
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/run` | Trigger agent loop |
| GET | `/status` | Poll run status |
| GET | `/report` | Get latest markdown report |

### Run with default dataset (Acocella 1984)

```bash
curl -X POST https://your-service.run.app/run
```

### Run with your own data

Upload any CSV with columns: `ID, TIME, CONC, Dose, Condition`

```bash
curl -X POST https://your-service.run.app/run \
  -F "data=@my_pk_data.csv"
```

### Poll and retrieve report

```bash
curl https://your-service.run.app/status
curl https://your-service.run.app/report
```

---

## Deploy to Cloud Run

```bash
gcloud run deploy pkpd-agent \
  --source . \
  --set-env-vars GOOGLE_API_KEY=your_key \
  --region europe-west1 \
  --allow-unauthenticated
```

---

## LLM providers

| Provider | Variable | Model |
|----------|----------|-------|
| Gemini (default) | `GOOGLE_API_KEY` | `gemini-1.5-flash` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-latest` |
| Local (HuggingFace) | — | see `LOCAL_MODEL_NAME` |

```bash
# Switch provider
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_key

# Local model
export LLM_PROVIDER=local
uv pip install -e ".[local]"
```

---

## Validation

```bash
python3 -m poc.validate
```

Outputs:
- `poc/validation_residuals.csv`
- `poc/validation_summary.md`
- `poc/plots/obs_vs_pred.png` (requires matplotlib)

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | `gemini`, `anthropic`, or `local` |
| `GOOGLE_API_KEY` | — | Required for Gemini |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic |
| `PARALLEL_AGENTS` | `0` | Run paper extraction in parallel with fitting |
| `GATE_MAX_RMSE` | — | Stop early if pooled RMSE exceeds threshold |
| `SMOKE_TEST` | `0` | Minimal run (1 subject, 5 obs) for CI |

---

## Project structure

```
poc/
  agent_poc.py   orchestration & entry point
  agents.py      agent definitions + shared AgentState
  model.py       PK math (predict, grid search)
  llm_utils.py   Gemini / Anthropic / local LLM integration
  io_utils.py    CSV parsing, metadata, I/O
  validate.py    residuals & plots
  app.py         FastAPI app (Cloud Run)
data/
  pkpd_acocella_1984_data.csv
  acocella_1984_metadata.json
  acocella_1984_paper.pdf
Dockerfile
```

---

## Limitations

- Single-compartment model only (no 2-cpt, no mixed effects)
- Grid search fitting (no confidence intervals)
- Single-user API (in-memory state)
