"""Combine two FR-B4 grid runs and re-derive L_critical on pooled seeds.

Usage:
    python analysis/combine_fr_b4_runs.py \
        results/FR-B4/fr_b4_full_grid_run1.json \
        results/FR-B4/fr_b4_full_grid_run2.json \
        --output results/FR-B4/fr_b4_combined.json

Each input JSON is the output of run_fr_b4_adaptive_coordination.py. The
combiner merges per-seed rows for every matching (omega, L, method) cell and
recomputes Δ̄, SD, sign_count, and n_seeds from the pooled set. L_critical
is then re-derived with the combined sign threshold (>=60% of pooled seeds).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


OMEGA_LABELS = ["stationary", "slow", "mid", "fast"]
L_LABELS     = ["L1", "L3", "L10", "L30", "Lall"]
METHODS      = ["window", "decay"]
LCRIT_FRAC   = 0.60   # same as LCRIT_SIGN_MIN / n_seeds in the runner


def _cell_stats(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [r["delta"] for r in per_seed]
    n = len(deltas)
    db = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    sc = sum(1 for d in deltas if d > 0)
    return {
        "delta_bar":  db,
        "delta_sd":   sd,
        "sign_count": sc,
        "n_seeds":    n,
        "per_seed":   per_seed,
    }


def _find_l_critical(omega_label: str, method: str, cells: dict[str, Any]) -> int | None:
    """Return smallest L with sign_count/n >= LCRIT_FRAC, or None."""
    for L_label in L_LABELS:
        key = f"{omega_label}|{L_label}|{method}"
        cell = cells.get(key)
        if cell is None:
            continue
        n = cell["n_seeds"]
        if n == 0:
            continue
        if cell["sign_count"] / n >= LCRIT_FRAC:
            return cell.get("L_steps", None) or int(L_label.lstrip("L").replace("all", "67"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine FR-B4 grid runs")
    parser.add_argument("run1", type=Path, help="First run JSON")
    parser.add_argument("run2", type=Path, help="Second run JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    r1 = json.loads(args.run1.read_text())
    r2 = json.loads(args.run2.read_text())

    # Index per-seed rows by (omega_label, L_label, method) from each run
    def index_cells(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        idx: dict[str, list[dict[str, Any]]] = {}
        for cell_entry in run.get("results", []):
            key = f"{cell_entry['omega_label']}|{cell_entry['L_label']}|{cell_entry['method']}"
            idx[key] = cell_entry.get("per_seed", [])
        return idx

    c1 = index_cells(r1)
    c2 = index_cells(r2)

    all_keys = sorted(set(c1) | set(c2))
    combined_cells: dict[str, Any] = {}
    for key in all_keys:
        pooled = c1.get(key, []) + c2.get(key, [])
        combined_cells[key] = _cell_stats(pooled)

    # L_critical per (omega, method)
    l_critical: dict[str, Any] = {}
    print("\nL_critical summary (combined, sign≥60%):")
    for omega_label in OMEGA_LABELS:
        for method in METHODS:
            lcrit = _find_l_critical(omega_label, method, combined_cells)
            l_critical[f"{omega_label}|{method}"] = lcrit
            print(f"  {omega_label:12s}  {method:6s}  L_critical={lcrit}")

    # Print full table
    print("\nFull combined results:")
    print(f"  {'omega':12s} {'L':5s} {'method':6s}  {'Δ̄':>7s}  {'SD':>5s}  sign/n")
    for omega_label in OMEGA_LABELS:
        for L_label in L_LABELS:
            for method in METHODS:
                key = f"{omega_label}|{L_label}|{method}"
                cell = combined_cells.get(key)
                if cell is None:
                    continue
                db = cell["delta_bar"]
                sd = cell["delta_sd"]
                sc = cell["sign_count"]
                n  = cell["n_seeds"]
                flag = " ✅" if n > 0 and sc / n >= LCRIT_FRAC else ""
                print(f"  {omega_label:12s} {L_label:5s} {method:6s}  {db:+7.3f}  {sd:5.3f}  {sc}/{n}{flag}")

    out = {
        "source_runs": [str(args.run1), str(args.run2)],
        "results": [
            {
                "omega_label": key.split("|")[0],
                "L_label":     key.split("|")[1],
                "method":      key.split("|")[2],
                **cell,
            }
            for key, cell in combined_cells.items()
        ],
        "l_critical": {k: v for k, v in l_critical.items()},
        "lcrit_criterion": f"sign_count / n_seeds >= {LCRIT_FRAC}",
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
