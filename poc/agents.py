from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from .io_utils import summarize_conditions, write_results, load_metadata
from .llm_utils import extract_paper_insights, extract_paper_insights_gemini, extract_paper_insights_local
from .model import Row, grid_search_bounded

# Initial search bounds
_CL_BOUNDS_DEFAULT: Tuple[float, float] = (0.1, 50.0)
_V_BOUNDS_DEFAULT: Tuple[float, float] = (1.0, 300.0)


@dataclass
class AgentState:
    """Shared state passed across agent steps."""
    rows: List[Row]
    by_subject: Dict[str, List[Row]]
    pooled_fit: Optional[Tuple[float, float, float]] = None
    results: Optional[List[Dict[str, float]]] = None
    last_rmse: Optional[float] = None
    no_improve_count: int = 0
    paper_insights: Optional[Dict[str, Any]] = None
    # Agentic loop state: enriched across iterations
    excluded_subjects: Set[str] = field(default_factory=set)
    grid_cl_bounds: Tuple[float, float] = field(default_factory=lambda: _CL_BOUNDS_DEFAULT)
    grid_v_bounds: Tuple[float, float] = field(default_factory=lambda: _V_BOUNDS_DEFAULT)


def agent_inspect(state: AgentState) -> None:
    """Flag outlier subjects whose RMSE > 2x the median across all subjects."""
    if not state.results:
        return
    rmses = [r["rmse"] for r in state.results]
    median = _median(rmses)
    threshold = median * 2.0
    newly_excluded = {
        str(r["subject_id"]) for r in state.results if r["rmse"] > threshold
    }
    state.excluded_subjects |= newly_excluded


def agent_fit_individual(state: AgentState) -> None:
    """Fit per-subject parameters using current grid bounds, skipping excluded subjects."""
    cl_min, cl_max = state.grid_cl_bounds
    v_min, v_max = state.grid_v_bounds
    results: List[Dict[str, float]] = []
    for subject_id, srows in sorted(state.by_subject.items()):
        if subject_id in state.excluded_subjects:
            continue
        sse, cl, v = grid_search_bounded(srows, cl_min, cl_max, v_min, v_max)
        rmse = math.sqrt(sse / max(len(srows), 1))
        results.append(
            {
                "subject_id": subject_id,
                "cl": cl,
                "v": v,
                "rmse": rmse,
                "n_obs": len(srows),
            }
        )
    state.results = results


def agent_fit_pooled(state: AgentState) -> None:
    """Fit pooled parameters on active subjects, then refine grid bounds for next iteration."""
    cl_min, cl_max = state.grid_cl_bounds
    v_min, v_max = state.grid_v_bounds
    active_rows = [r for r in state.rows if r.subject_id not in state.excluded_subjects]
    if not active_rows:
        return
    state.pooled_fit = grid_search_bounded(active_rows, cl_min, cl_max, v_min, v_max)
    # Refine grid bounds around best (CL, V) for next iteration
    _, best_cl, best_v = state.pooled_fit
    state.grid_cl_bounds = _zoom_bounds(best_cl, cl_min, cl_max)
    state.grid_v_bounds = _zoom_bounds(best_v, v_min, v_max)


def agent_report(state: AgentState, out_results: Path, out_report: Path, meta_path: Path) -> None:
    """Generate CSV results and a short markdown report."""
    if not state.pooled_fit or not state.results:
        return
    write_results(out_results, state.results)
    write_report(
        out_report,
        meta_path,
        state.rows,
        state.pooled_fit,
        state.results,
        state.paper_insights,
        state.excluded_subjects,
    )


def agent_read_paper(
    state: AgentState, pdf_path: Path, api_key: str, model_name: str, provider: str
) -> None:
    """Extract key model details from the paper using an LLM."""
    if state.paper_insights is not None:
        return
    if provider == "anthropic":
        if not api_key:
            return
        state.paper_insights = extract_paper_insights(pdf_path, api_key, model_name)
        return
    if provider == "gemini":
        if not api_key:
            return
        state.paper_insights = extract_paper_insights_gemini(pdf_path, api_key, model_name)
        return
    if provider == "local":
        state.paper_insights = extract_paper_insights_local(pdf_path, model_name)
        return


def write_report(
    out_path: Path,
    meta_path: Path,
    rows: List[Row],
    pooled_fit: Tuple[float, float, float],
    results: List[Dict[str, Any]],
    paper_insights: Optional[Dict[str, Any]],
    excluded_subjects: Optional[Set[str]] = None,
) -> None:
    """Write a minimal markdown report for the POC run."""
    meta = load_metadata(meta_path)
    n_obs = len(rows)
    n_subjects = len({r.subject_id for r in rows})
    times = sorted({r.time for r in rows})
    doses, tinfs = summarize_conditions(rows)
    pooled_sse, pooled_cl, pooled_v = pooled_fit
    pooled_rmse = math.sqrt(pooled_sse / max(n_obs, 1))
    best_subjects = sorted(results, key=lambda x: x["rmse"])[:5]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# PK Agent POC Report (Acocella 1984)\n\n")
        f.write("## Data summary\n")
        f.write(f"- Subjects: {n_subjects}\n")
        f.write(f"- Observations: {n_obs}\n")
        f.write(f"- Time range (h): {min(times)} to {max(times)}\n")
        f.write(f"- Doses (mg): {', '.join(str(d) for d in doses)}\n")
        f.write(f"- Infusion durations (h): {', '.join(str(t) for t in tinfs)}\n")
        if meta:
            f.write(f"- Units: time={meta.get('time_unit')}, conc={meta.get('conc_unit')}\n")
        f.write("\n")

        f.write("## Model\n")
        f.write("- One-compartment IV infusion with first-order elimination\n")
        f.write("- Parameters: CL, V, k = CL/V\n\n")

        f.write("## Results (POC grid search)\n")
        f.write(f"- Pooled fit: CL={pooled_cl:.3g}, V={pooled_v:.3g}, RMSE={pooled_rmse:.3g}\n")
        f.write("- Best individual fits (lowest RMSE):\n")
        for r in best_subjects:
            f.write(
                f"  - Subject {r['subject_id']}: CL={r['cl']:.3g}, V={r['v']:.3g}, RMSE={r['rmse']:.3g}\n"
            )
        f.write("\n")

        f.write("## Limitations\n")
        f.write("- No population model (no mixed effects)\n")
        f.write("- Coarse grid search only (no CI, no diagnostics)\n")
        f.write("- No plots in this POC\n")

        if excluded_subjects:
            f.write("\n## Outliers excluded\n")
            f.write(f"- Subjects removed (RMSE > 2× median): {', '.join(sorted(excluded_subjects))}\n")

        if paper_insights:
            f.write("\n## Paper-derived model notes (LLM)\n")
            for key, val in paper_insights.items():
                f.write(f"- {key}: {val}\n")


def _median(values: List[float]) -> float:
    """Return the median of a non-empty list."""
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _zoom_bounds(best: float, lo: float, hi: float) -> Tuple[float, float]:
    """Halve the log-space range around best, staying within [lo, hi]."""
    log_lo = math.log10(lo)
    log_hi = math.log10(hi)
    log_best = math.log10(max(best, 1e-9))
    half_range = (log_hi - log_lo) / 4.0  # halve the range each iteration
    new_lo = 10 ** max(log_lo, log_best - half_range)
    new_hi = 10 ** min(log_hi, log_best + half_range)
    return (new_lo, new_hi)


def should_stop(state: AgentState, iteration: int, max_iter: int, no_improve_stop: int) -> bool:
    """Stop when max_iter reached or no improvement for no_improve_stop iterations."""
    if iteration >= max_iter:
        return True
    if state.pooled_fit is None:
        return False
    pooled_sse, _, _ = state.pooled_fit
    pooled_rmse = math.sqrt(pooled_sse / max(len(state.rows), 1))
    if state.last_rmse is None or pooled_rmse < state.last_rmse:
        state.last_rmse = pooled_rmse
        state.no_improve_count = 0
        return False
    state.no_improve_count += 1
    return state.no_improve_count >= no_improve_stop
