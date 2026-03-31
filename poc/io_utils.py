from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .model import Row


def safe_float(value: str) -> float:
    """Parse a float from a string, accepting comma decimals."""
    value = value.strip().replace(",", ".")
    return float(value)


def parse_infusion_duration(condition: str) -> float:
    """Infer infusion duration (hours) from the Condition string."""
    text = condition.lower()
    if "infusion" not in text:
        return 0.0
    for token in text.replace("(", " ").replace(")", " ").split(","):
        token = token.strip()
        if "h infusion" in token or "hour infusion" in token:
            num = token.split(" ")[0]
            return safe_float(num)
    idx = text.find("h")
    if idx > 0:
        num = text[:idx].split()[-1]
        try:
            return safe_float(num)
        except ValueError:
            return 0.0
    return 0.0


def read_rows(csv_path: Path) -> List[Row]:
    """Read raw CSV into typed Row objects."""
    rows: List[Row] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            subject_id = r.get("ID") or r.get("participant_id") or ""
            time = safe_float(r.get("TIME") or r.get("Time") or "0")
            conc = safe_float(r.get("CONC") or r.get("Avg") or "0")
            dose_mg = safe_float(r.get("Dose") or "0")
            condition = r.get("Condition") or ""
            tinf_h = parse_infusion_duration(condition)
            rows.append(
                Row(
                    subject_id=str(subject_id),
                    time=time,
                    conc=conc,
                    dose_mg=dose_mg,
                    condition=condition,
                    tinf_h=tinf_h,
                )
            )
    return rows


def summarize_conditions(rows: List[Row]) -> Tuple[List[float], List[float]]:
    """Return unique doses and infusion durations."""
    doses = sorted({r.dose_mg for r in rows})
    tinfs = sorted({r.tinf_h for r in rows})
    return doses, tinfs


def load_metadata(meta_path: Path) -> Dict[str, Any]:
    """Load metadata JSON if it exists."""
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_results(out_path: Path, results: List[Dict[str, float]]) -> None:
    """Write per-subject fit results to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys()) if results else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
