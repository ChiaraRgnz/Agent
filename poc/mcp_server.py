"""
MCP server exposing PK analysis tools.

Each tool is a discrete action the LLM orchestrator can call.
State persists across tool calls within a session (module-level).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from .agents import _zoom_bounds, write_report
from .io_utils import read_rows, write_results, load_metadata
from .llm_utils import extract_paper_insights, extract_paper_insights_gemini
from .model import Row, grid_search_bounded, group_by_subject

DATA_CSV = Path("data/pkpd_acocella_1984_data.csv")
PAPER_PDF = Path("data/acocella_1984_paper.pdf")
META_JSON = Path("data/acocella_1984_metadata.json")
OUT_RESULTS = Path("poc/results.csv")
OUT_REPORT = Path("poc/report.md")

mcp = FastMCP("pkpd-agent")

# Session state — persists across tool calls
_rows: List[Row] = []
_by_subject: Dict[str, List[Row]] = {}
_results: List[Dict[str, Any]] = []
_pooled_fit: tuple = ()
_excluded: set = set()
_paper_insights: Dict[str, Any] = {}
_grid_cl: tuple = (0.1, 50.0)
_grid_v: tuple = (1.0, 300.0)


def _ensure_data(csv_path: Path = DATA_CSV) -> None:
    global _rows, _by_subject
    if not _rows:
        _rows = read_rows(csv_path)
        _by_subject = group_by_subject(_rows)


@mcp.tool()
def get_data_summary() -> Dict[str, Any]:
    """Return summary statistics of the PK dataset: subjects, observations, doses, time range."""
    _ensure_data()
    times = sorted({r.time for r in _rows})
    doses = sorted({r.dose_mg for r in _rows})
    return {
        "n_subjects": len(_by_subject),
        "n_obs": len(_rows),
        "time_range_h": [min(times), max(times)],
        "doses_mg": doses,
        "excluded_subjects": list(_excluded),
    }


@mcp.tool()
def run_individual_fit(exclude: List[str] = []) -> Dict[str, Any]:
    """
    Fit CL and V per subject via grid search.
    Returns RMSE per subject — use this to identify outliers before pooled fit.
    """
    global _results, _excluded
    _ensure_data()
    _excluded = set(exclude)
    cl_min, cl_max = _grid_cl
    v_min, v_max = _grid_v
    results = []
    for sid, srows in sorted(_by_subject.items()):
        if sid in _excluded:
            continue
        sse, cl, v = grid_search_bounded(srows, cl_min, cl_max, v_min, v_max)
        rmse = math.sqrt(sse / max(len(srows), 1))
        results.append({
            "subject_id": sid,
            "cl": round(cl, 3),
            "v": round(v, 3),
            "rmse": round(rmse, 3),
            "n_obs": len(srows),
        })
    _results = results
    rmses = [r["rmse"] for r in results]
    median = sorted(rmses)[len(rmses) // 2]
    return {
        "n_fitted": len(results),
        "rmse_min": round(min(rmses), 3),
        "rmse_median": round(median, 3),
        "rmse_max": round(max(rmses), 3),
        "outlier_threshold": round(median * 2, 3),
        "results": results,
    }


@mcp.tool()
def run_pooled_fit(exclude: List[str] = []) -> Dict[str, Any]:
    """
    Fit a single CL and V across all active subjects.
    Also refines the grid bounds for subsequent calls.
    """
    global _pooled_fit, _excluded, _grid_cl, _grid_v
    _ensure_data()
    _excluded = set(exclude)
    active = [r for r in _rows if r.subject_id not in _excluded]
    cl_min, cl_max = _grid_cl
    v_min, v_max = _grid_v
    sse, cl, v = grid_search_bounded(active, cl_min, cl_max, v_min, v_max)
    rmse = math.sqrt(sse / max(len(active), 1))
    _pooled_fit = (sse, cl, v)
    _grid_cl = _zoom_bounds(cl, cl_min, cl_max)
    _grid_v = _zoom_bounds(v, v_min, v_max)
    return {
        "cl": round(cl, 3),
        "v": round(v, 3),
        "rmse": round(rmse, 3),
        "n_obs": len(active),
        "excluded": list(_excluded),
    }


@mcp.tool()
def extract_paper() -> Dict[str, Any]:
    """Extract model details from the reference PDF using an LLM. Cached after first call."""
    global _paper_insights
    if _paper_insights:
        return _paper_insights
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    api_key = os.environ.get("GOOGLE_API_KEY", "") if provider == "gemini" else os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "no_api_key"}
    if provider == "gemini":
        _paper_insights = extract_paper_insights_gemini(PAPER_PDF, api_key)
    else:
        _paper_insights = extract_paper_insights(PAPER_PDF, api_key, "claude-3-5-sonnet-latest")
    return _paper_insights


@mcp.tool()
def generate_report() -> str:
    """Write the final CSV and markdown report. Call this only after fits are complete."""
    if not _results or not _pooled_fit:
        return "error: run fits before generating report"
    write_results(OUT_RESULTS, _results)
    write_report(OUT_REPORT, META_JSON, _rows, _pooled_fit, _results, _paper_insights or None, _excluded)
    return f"report written to {OUT_REPORT}"
