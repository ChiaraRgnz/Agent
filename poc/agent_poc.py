#!/usr/bin/env python3
"""
Minimal PK agent POC (stdlib only).
Orchestrates a multi-agent loop with stop criteria.
"""

from __future__ import annotations

from pathlib import Path
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .agents import (
    AgentState,
    agent_fit_individual,
    agent_fit_pooled,
    agent_inspect,
    agent_read_paper,
    agent_report,
    should_stop,
)
from .io_utils import read_rows
from .model import Row, group_by_subject


DATA_CSV = Path("data/pkpd_acocella_1984_data.csv")
META_JSON = Path("data/acocella_1984_metadata.json")
OUT_RESULTS = Path("poc/results.csv")
OUT_REPORT = Path("poc/report.md")
PAPER_PDF = Path("data/acocella_1984_paper.pdf")

MAX_ITERATIONS = 5
NO_IMPROVEMENT_STOP = 2
ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
GEMINI_MODEL = "gemini-1.5-flash"
LOCAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def run_agent_loop(state: AgentState) -> None:
    """Run the agent loop with stop criteria."""
    agents = [agent_inspect, agent_fit_individual, agent_fit_pooled]
    iteration = 0
    while True:
        provider = _llm_provider()
        if provider == "anthropic":
            model_name = ANTHROPIC_MODEL
        elif provider == "gemini":
            model_name = GEMINI_MODEL
        else:
            model_name = _local_model()
        paper_future = None
        if _parallel_enabled():
            with ThreadPoolExecutor(max_workers=2) as pool:
                paper_future = pool.submit(
                    agent_read_paper,
                    state,
                    PAPER_PDF,
                    _llm_api_key(),
                    model_name,
                    provider,
                )
                for agent in agents:
                    agent(state)
                paper_future.result()
            if not _gate_passed(state):
                break
        else:
            for agent in agents:
                agent(state)
            agent_read_paper(state, PAPER_PDF, _llm_api_key(), model_name, provider)
        agent_report(state, OUT_RESULTS, OUT_REPORT, META_JSON)
        if should_stop(state, iteration, MAX_ITERATIONS, NO_IMPROVEMENT_STOP):
            break
        iteration += 1


def main() -> None:
    """Entry point: orchestrated (MCP) or fixed pipeline depending on USE_ORCHESTRATOR."""
    if _orchestrator_enabled():
        from .orchestrator import run_orchestrated
        run_orchestrated(_anthropic_key())
        return
    rows = read_rows(DATA_CSV)
    if _smoke_test_enabled():
        rows = _smoke_subset(rows)
    by_subject = group_by_subject(rows)
    state = AgentState(rows=rows, by_subject=by_subject)
    run_agent_loop(state)


def _anthropic_key() -> str:
    """Read Anthropic API key from environment."""
    import os

    return os.environ.get("ANTHROPIC_API_KEY", "")


def _llm_api_key() -> str:
    """Return the API key for the active LLM provider."""
    import os

    provider = _llm_provider()
    if provider == "gemini":
        return os.environ.get("GOOGLE_API_KEY", "")
    return _anthropic_key()


def _llm_provider() -> str:
    """Return the LLM provider: 'anthropic' or 'local'."""
    import os

    return os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()


def _local_model() -> str:
    """Read local model name from env or fallback default."""
    import os

    return os.environ.get("LOCAL_MODEL_NAME", LOCAL_MODEL)


def _smoke_test_enabled() -> bool:
    """Return True if SMOKE_TEST is enabled."""
    import os

    return os.environ.get("SMOKE_TEST", "0").strip() == "1"


def _smoke_subset(rows: List[Row]) -> List[Row]:
    """Return a tiny subset for low-resource smoke testing."""
    # Pick first subject and first 5 observations.
    if not rows:
        return rows
    first_id = rows[0].subject_id
    subset = [r for r in rows if r.subject_id == first_id][:5]
    return subset


def _parallel_enabled() -> bool:
    """Return True if PARALLEL_AGENTS is enabled."""
    import os

    return os.environ.get("PARALLEL_AGENTS", "0").strip() == "1"


def _orchestrator_enabled() -> bool:
    """Return True if USE_ORCHESTRATOR is enabled."""
    import os

    return os.environ.get("USE_ORCHESTRATOR", "0").strip() == "1"


def _gate_passed(state: AgentState) -> bool:
    """Inline gate: only proceed if pooled RMSE <= threshold."""
    threshold = _gate_max_rmse()
    if threshold is None:
        return True
    if state.pooled_fit is None:
        return False
    pooled_sse, _, _ = state.pooled_fit
    pooled_rmse = math.sqrt(pooled_sse / max(len(state.rows), 1))
    return pooled_rmse <= threshold


def _gate_max_rmse() -> float | None:
    """Read RMSE gate threshold from env, or None if unset."""
    import os

    val = os.environ.get("GATE_MAX_RMSE", "").strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
