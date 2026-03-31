from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class Row:
    """Single observation row after parsing raw CSV."""
    subject_id: str
    time: float
    conc: float
    dose_mg: float
    condition: str
    tinf_h: float


def logspace(min_val: float, max_val: float, n: int) -> List[float]:
    """Generate n log-spaced values between min_val and max_val (inclusive)."""
    if n <= 1:
        return [min_val]
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)
    step = (log_max - log_min) / (n - 1)
    return [10 ** (log_min + i * step) for i in range(n)]


def predict_conc(time_h: float, dose_mg: float, tinf_h: float, cl: float, v: float) -> float:
    """Predict concentration for a 1-compartment IV infusion model."""
    if cl <= 0 or v <= 0:
        return 0.0
    k = cl / v
    if tinf_h <= 0:
        return (dose_mg / v) * math.exp(-k * time_h)
    r0 = dose_mg / tinf_h
    if time_h <= tinf_h:
        return (r0 / cl) * (1.0 - math.exp(-k * time_h))
    return (r0 / cl) * (1.0 - math.exp(-k * tinf_h)) * math.exp(-k * (time_h - tinf_h))


def sse_for(rows: Iterable[Row], cl: float, v: float) -> float:
    """Compute sum of squared errors for a set of rows and parameters."""
    err = 0.0
    for r in rows:
        pred = predict_conc(r.time, r.dose_mg, r.tinf_h, cl, v)
        diff = r.conc - pred
        err += diff * diff
    return err


def grid_search(rows: List[Row]) -> Tuple[float, float, float]:
    """Coarse grid-search fit returning (sse, cl, v)."""
    return grid_search_bounded(rows, 0.1, 50.0, 1.0, 300.0)


def grid_search_bounded(
    rows: List[Row],
    cl_min: float,
    cl_max: float,
    v_min: float,
    v_max: float,
    n: int = 25,
) -> Tuple[float, float, float]:
    """Grid-search fit within explicit CL and V bounds, returning (sse, cl, v)."""
    cl_grid = logspace(cl_min, cl_max, n)
    v_grid = logspace(v_min, v_max, n)
    best = (float("inf"), 0.0, 0.0)
    for cl in cl_grid:
        for v in v_grid:
            sse = sse_for(rows, cl, v)
            if sse < best[0]:
                best = (sse, cl, v)
    return best


def group_by_subject(rows: List[Row]) -> Dict[str, List[Row]]:
    """Group rows by subject_id."""
    out: Dict[str, List[Row]] = {}
    for r in rows:
        out.setdefault(r.subject_id, []).append(r)
    return out
