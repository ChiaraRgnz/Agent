#!/usr/bin/env python3
"""
FastAPI wrapper for the PKPD agent — deployable on Cloud Run.

Environment variables:
  LLM_PROVIDER       anthropic | gemini | local  (default: gemini)
  GOOGLE_API_KEY     required for gemini
  ANTHROPIC_API_KEY  required for anthropic
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse

from .agent_poc import run_agent_loop, _llm_provider, _llm_api_key
from .io_utils import read_rows
from .model import group_by_subject
from .agents import AgentState

DATA_CSV = Path("data/pkpd_acocella_1984_data.csv")
OUT_REPORT = Path("poc/report.md")
META_JSON = Path("data/acocella_1984_metadata.json")

app = FastAPI(
    title="PKPD Agent",
    description="Agentic PK modelling — one-compartment IV infusion",
    version="0.1.0",
)

# In-memory run status (single-user POC)
_status: dict = {"state": "idle", "error": None}


def _run(csv_path: Path) -> None:
    _status["state"] = "running"
    _status["error"] = None
    try:
        rows = read_rows(csv_path)
        by_subject = group_by_subject(rows)
        state = AgentState(rows=rows, by_subject=by_subject)
        run_agent_loop(state)
        _status["state"] = "done"
    except Exception as exc:
        _status["state"] = "error"
        _status["error"] = str(exc)


@app.get("/health")
def health() -> dict:
    """Health check for Cloud Run."""
    return {"status": "ok", "llm_provider": _llm_provider()}


@app.post("/run")
async def run(
    background_tasks: BackgroundTasks,
    data: Optional[UploadFile] = File(default=None),
) -> dict:
    """
    Trigger the agent loop asynchronously.

    - Without file: uses the bundled Acocella 1984 dataset.
    - With file: upload your own CSV (columns: ID, TIME, CONC, Dose, Condition).
    """
    if _status["state"] == "running":
        raise HTTPException(status_code=409, detail="Agent already running")

    if data is not None:
        # Write upload to a temp file so the background task can read it
        suffix = Path(data.filename or "data.csv").suffix or ".csv"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await data.read())
        tmp.flush()
        csv_path = Path(tmp.name)
    else:
        csv_path = DATA_CSV

    background_tasks.add_task(_run, csv_path)
    return {
        "status": "started",
        "provider": _llm_provider(),
        "dataset": data.filename if data else "acocella_1984 (default)",
    }


@app.get("/status")
def status() -> dict:
    """Return current run status."""
    return _status


@app.get("/report", response_class=PlainTextResponse)
def report() -> str:
    """Return the latest markdown report."""
    if not OUT_REPORT.exists():
        raise HTTPException(status_code=404, detail="No report yet — run POST /run first")
    return OUT_REPORT.read_text(encoding="utf-8")
