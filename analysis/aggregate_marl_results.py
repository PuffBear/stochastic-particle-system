"""Aggregate per-seed JSON results from SPS-WO-08 HPC jobs into one report.

After all 48 PBS jobs complete (6 archs × 8 seeds), run:

    python analysis/aggregate_marl_results.py \\
        --input-dir results/raw/SPS-WO-08-MARL \\
        --output results/sps-wo-08-marl-baselines-aggregate.json

The script:
  1. Discovers all per-seed JSONs matching {arch}_seed{seed}.json
  2. Validates seed firewalls and experiment IDs
  3. Aggregates mean/std yield across 8 train seeds per architecture
  4. Computes per-arch mean over the 8 eval seeds (mean-of-means)
  5. Prints a ranked comparison table against scripted baselines
  6. Writes a self-contained aggregate JSON
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ID = "SPS-WO-08-MARL-BASELINES"
ALL_ARCHS = ("ippo", "mappo", "commnet", "coma", "vdn", "maddpg")
TRAIN_SEED_RANGE = range(8001, 8009)
EVAL_SEEDS = tuple(range(9001, 9009))


def load_results(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all per-seed JSON files, grouped by architecture."""
    by_arch: dict[str, list[dict[str, Any]]] = {a: [] for a in ALL_ARCHS}
    missing: list[str] = []
    extra: list[str] = []

    for arch in ALL_ARCHS:
        for seed in TRAIN_SEED_RANGE:
            path = input_dir / f"{arch}_seed{seed}.json"
            if path.exists():
                data = json.loads(path.read_text())
                if data.get("experiment_id") != EXPERIMENT_ID:
                    raise ValueError(
                        f"{path}: experiment_id mismatch — "
                        f"expected {EXPERIMENT_ID!r}, got {data.get('experiment_id')!r}"
                    )
                if data.get("train_seed") != seed:
                    raise ValueError(
                        f"{path}: train_seed mismatch — expected {seed}, got {data.get('train_seed')}"
                    )
                by_arch[arch].append(data)
            else:
                missing.append(f"{arch}_seed{seed}")

    if missing:
        print(f"WARNING: {len(missing)} result file(s) missing:")
        for m in missing:
            print(f"  {m}.json")

    return by_arch


def arch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise one architecture across all its per-seed records."""
    if not records:
        return {"n_seeds": 0, "status": "no_results"}

    seed_means = [r["learned_eval"]["mean_yield"] for r in records]
    # Each per-seed record already has mean over 8 eval seeds.
    return {
        "n_seeds": len(records),
        "seed_means": seed_means,
        "mean_over_seeds": float(np.mean(seed_means)),
        "std_over_seeds": float(np.std(seed_means)),
        "min_seed_mean": float(np.min(seed_means)),
        "max_seed_mean": float(np.max(seed_means)),
        "wall_time_s_per_seed": [r.get("wall_time_s") for r in records],
    }


def scripted_summary(records: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    """Extract scripted baseline means from the per-seed records (all should agree)."""
    means = []
    for r in records:
        baseline = r.get("scripted_baselines", {}).get(policy_name)
        if baseline is not None:
            means.append(baseline["mean_yield"])
    if not means:
        return {"mean_yield": None, "n_records": 0}
    return {
        "mean_yield": float(np.mean(means)),
        "std_yield": float(np.std(means)),
        "n_records": len(means),
    }


def print_table(arch_summaries: dict[str, Any], scripted: dict[str, Any]) -> None:
    print(f"\n{'='*65}")
    print("SPS-WO-08  MARL vs Scripted Baselines  (mean yield over 8 eval seeds)")
    print(f"{'='*65}")
    print(f"  {'Policy / Architecture':<40s}  {'Mean':>7s}  {'±Std':>7s}  {'N':>3s}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*7}  {'-'*3}")

    scripted_order = [
        ("stationary", "stationary"),
        ("capacity_matched_independent", "capacity_matched_independent"),
        ("shared_summary_v2", "shared_summary_v2 (SPS-C03)"),
        ("full_state_interception_oracle", "oracle (ceiling)"),
    ]
    for key, label in scripted_order:
        s = scripted.get(key, {})
        mean = s.get("mean_yield")
        std = s.get("std_yield", 0.0)
        if mean is not None:
            print(f"  {label:<40s}  {mean:>7.4f}  {std:>7.4f}   --")

    print(f"  {'- - MARL - -':<40s}")
    arch_rows = []
    for arch in ALL_ARCHS:
        s = arch_summaries.get(arch, {})
        if s.get("n_seeds", 0) > 0:
            arch_rows.append((s["mean_over_seeds"], arch, s))

    for _, arch, s in sorted(arch_rows, key=lambda x: -x[0]):
        print(
            f"  {arch:<40s}  "
            f"{s['mean_over_seeds']:>7.4f}  "
            f"{s['std_over_seeds']:>7.4f}  "
            f"{s['n_seeds']:>3d}"
        )
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Aggregate {EXPERIMENT_ID} per-seed results")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw/SPS-WO-08-MARL"),
        help="Directory containing per-seed JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sps-wo-08-marl-baselines-aggregate.json"),
        help="Path to write the aggregate JSON report.",
    )
    args = parser.parse_args()

    print(f"Loading per-seed results from {args.input_dir} ...")
    by_arch = load_results(args.input_dir)

    total = sum(len(v) for v in by_arch.values())
    print(f"Loaded {total} result file(s) across {len(ALL_ARCHS)} architectures.")

    arch_summaries = {arch: arch_summary(records) for arch, records in by_arch.items()}

    # Collect scripted baselines (any one arch's records will do — they should match)
    scripted: dict[str, Any] = {}
    all_records: list[dict] = [r for recs in by_arch.values() for r in recs]
    for pol in ("stationary", "capacity_matched_independent", "shared_summary_v2", "full_state_interception_oracle"):
        scripted[pol] = scripted_summary(all_records, pol)

    print_table(arch_summaries, scripted)

    aggregate: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_results_loaded": total,
        "eval_seeds": list(EVAL_SEEDS),
        "train_seeds": list(TRAIN_SEED_RANGE),
        "arch_summaries": arch_summaries,
        "scripted_baselines": scripted,
        "per_arch_raw": {arch: [
            {
                "train_seed": r["train_seed"],
                "mean_yield": r["learned_eval"]["mean_yield"],
                "std_yield": r["learned_eval"]["std_yield"],
                "wall_time_s": r.get("wall_time_s"),
            }
            for r in records
        ] for arch, records in by_arch.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2))
    print(f"Aggregate report written to {args.output}")


if __name__ == "__main__":
    main()
