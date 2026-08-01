#!/usr/bin/env python3
"""Frozen design-stage power analysis for SPS-C03 confirmation.

This script consumes no diagnostic outcomes.  It sizes a one-sided paired
studentized-bootstrap confirmation test for an effect frozen ex ante at two
additional unique team captures.  Outputs are immutable and provenance-rich.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import yaml

from particle_benchmark.io import canonical_json_bytes


COORDINATION_MINIMUM_EFFECT = 2.0
ASSUMED_SD_RANGE = (2.0, 4.0)
ASSUMED_SD_STEPS = 5
SEED_COUNTS = (16, 24, 32)
BOOTSTRAP_DRAWS = 10_000
CALIBRATION_DATASETS = 100
SIMULATION_TRIALS = 2_000
SIMULATION_RNG_SEED = 8_421
TARGET_POWER = 0.80
ONE_SIDED_ALPHA = 0.05
EXPERIMENT_ID = "SPS-POWER-ANALYSIS-COORDINATION"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calibrate_critical_value(
    n_seeds: int,
    *,
    bootstrap_draws: int,
    calibration_datasets: int,
    rng: np.random.Generator,
) -> float:
    """Calibrate the one-sided studentized-bootstrap critical value."""
    if n_seeds < 2 or bootstrap_draws < 1 or calibration_datasets < 1:
        raise ValueError("power calibration sizes must be positive and n_seeds >= 2")
    critical_values = np.empty(calibration_datasets)
    for index in range(calibration_datasets):
        data = rng.standard_normal(n_seeds)
        observed_mean = float(np.mean(data))
        bootstrap_indices = rng.integers(0, n_seeds, size=(bootstrap_draws, n_seeds))
        bootstrap_data = data[bootstrap_indices]
        bootstrap_means = np.mean(bootstrap_data, axis=1)
        bootstrap_ses = np.std(bootstrap_data, axis=1, ddof=1) / np.sqrt(n_seeds)
        shortfall = np.divide(
            observed_mean - bootstrap_means,
            bootstrap_ses,
            out=np.zeros_like(bootstrap_means),
            where=bootstrap_ses > 1e-15,
        )
        critical_values[index] = np.quantile(
            shortfall, 1.0 - ONE_SIDED_ALPHA, method="higher"
        )
    return float(np.median(critical_values))


def _simulate_power(
    n_seeds: int,
    assumed_sd: float,
    effect: float,
    critical_value: float,
    *,
    simulation_trials: int,
    rng: np.random.Generator,
) -> float:
    """Estimate rejection probability for paired seed-level differences."""
    if assumed_sd <= 0 or effect <= 0 or simulation_trials < 1:
        raise ValueError("effect, SD, and trial count must be positive")
    paired_differences = rng.normal(
        loc=effect, scale=assumed_sd, size=(simulation_trials, n_seeds)
    )
    means = np.mean(paired_differences, axis=1)
    standard_errors = np.std(paired_differences, axis=1, ddof=1) / np.sqrt(n_seeds)
    lower_bounds = means - critical_value * standard_errors
    return float(np.mean(lower_bounds > 0.0))


def build_power_report(
    *,
    assumed_sd_values: np.ndarray,
    seed_counts: tuple[int, ...],
    bootstrap_draws: int,
    calibration_datasets: int,
    simulation_trials: int,
    rng_seed: int,
) -> dict[str, object]:
    """Return the frozen report without reading any scientific outcomes."""
    rng = np.random.default_rng(rng_seed)
    calibrated: dict[int, float] = {}
    for n_seeds in seed_counts:
        calibrated[n_seeds] = _calibrate_critical_value(
            n_seeds,
            bootstrap_draws=bootstrap_draws,
            calibration_datasets=calibration_datasets,
            rng=rng,
        )

    rows: list[dict[str, object]] = []
    for n_seeds in seed_counts:
        for assumed_sd in assumed_sd_values:
            rows.append(
                {
                    "n_seeds": n_seeds,
                    "assumed_sd": float(assumed_sd),
                    "effect": COORDINATION_MINIMUM_EFFECT,
                    "calibrated_critical_value": calibrated[n_seeds],
                    "power": _simulate_power(
                        n_seeds,
                        float(assumed_sd),
                        COORDINATION_MINIMUM_EFFECT,
                        calibrated[n_seeds],
                        simulation_trials=simulation_trials,
                        rng=rng,
                    ),
                }
            )

    recommendations: list[dict[str, object]] = []
    for assumed_sd in assumed_sd_values:
        eligible = [
            row
            for row in rows
            if float(row["assumed_sd"]) == float(assumed_sd)
            and float(row["power"]) >= TARGET_POWER
        ]
        recommendations.append(
            {
                "assumed_sd": float(assumed_sd),
                "min_seeds_for_80pct_power": (
                    min(int(row["n_seeds"]) for row in eligible) if eligible else None
                ),
            }
        )

    if all(row["min_seeds_for_80pct_power"] is not None for row in recommendations):
        recommended_n: int | None = max(
            int(row["min_seeds_for_80pct_power"]) for row in recommendations
        )
        note = "Use the worst-case qualifying candidate count across the frozen SD range."
    else:
        recommended_n = None
        note = (
            "At least one frozen SD scenario does not reach 80% power within "
            "{16,24,32}; register a successor power design before enlarging the budget."
        )

    return {
        "work_order_id": "SPS-WO-08",
        "experiment_id": EXPERIMENT_ID,
        "analysis_type": "design-stage simulation; no SPS-C03 outcomes consumed",
        "coordination_minimum_effect": COORDINATION_MINIMUM_EFFECT,
        "minimum_effect_frozen_ex_ante": True,
        "minimum_effect_rationale": (
            "Two unique team captures equals 0.5 capture per collector and is the "
            "smallest benefit judged worth a confirmation battery. It is fixed before "
            "seeds 4001-4008 are observed and will not be derived from their outcomes."
        ),
        "assumed_sd_range": list(ASSUMED_SD_RANGE),
        "assumed_sd_values": assumed_sd_values.astype(float).tolist(),
        "assumed_sd_rationale": (
            "The range 2.0-4.0 brackets the earlier oracle-minus-stationary diagnostic "
            "SD of 2.825 without treating that different contrast as an estimate."
        ),
        "seed_counts_evaluated": list(seed_counts),
        "target_power": TARGET_POWER,
        "test": "one-sided paired studentized-bootstrap 95% lower bound > 0",
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "bootstrap_draws": bootstrap_draws,
        "calibration_datasets": calibration_datasets,
        "simulation_trials": simulation_trials,
        "simulation_rng_seed": rng_seed,
        "calibrated_critical_values": {str(key): value for key, value in calibrated.items()},
        "power_table": rows,
        "recommendations_by_sd": recommendations,
        "recommended_confirmation_seed_count": recommended_n,
        "recommendation_note": note,
    }


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-base-commit", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    if len(args.repository_base_commit) != 40:
        raise ValueError("repository base commit must be a full 40-character SHA")

    frozen = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    expected = {
        "minimum_relevant_effect": COORDINATION_MINIMUM_EFFECT,
        "assumed_sd_range": list(ASSUMED_SD_RANGE),
        "assumed_sd_steps": ASSUMED_SD_STEPS,
        "candidate_seed_counts": list(SEED_COUNTS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "calibration_datasets": CALIBRATION_DATASETS,
        "simulation_trials": SIMULATION_TRIALS,
        "simulation_rng_seed": SIMULATION_RNG_SEED,
        "target_power": TARGET_POWER,
        "one_sided_alpha": ONE_SIDED_ALPHA,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"frozen config {key} differs from SPS-WO-08 constants")

    started = time.perf_counter()
    report = build_power_report(
        assumed_sd_values=np.linspace(*ASSUMED_SD_RANGE, ASSUMED_SD_STEPS),
        seed_counts=SEED_COUNTS,
        bootstrap_draws=BOOTSTRAP_DRAWS,
        calibration_datasets=CALIBRATION_DATASETS,
        simulation_trials=SIMULATION_TRIALS,
        rng_seed=SIMULATION_RNG_SEED,
    )
    args.output.mkdir(parents=False)
    report_path = args.output / "power_report.json"
    _write_new(report_path, canonical_json_bytes(report))
    root = Path(__file__).resolve().parents[1]
    manifest = {
        "work_order_id": "SPS-WO-08",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_outcomes_consumed": False,
        "repository": {
            "full_name": "PuffBear/stochastic-particle-system",
            "branch": "research-autonomy",
            "base_commit_sha": args.repository_base_commit,
            "workspace_changes_included": True,
        },
        "source": {
            "path": "analysis/run_power_analysis.py",
            "sha256": _file_sha256(root / "analysis" / "run_power_analysis.py"),
        },
        "frozen_config": {
            "path": args.config.as_posix(),
            "sha256": _file_sha256(args.config),
            "contents": frozen,
        },
        "runtime": {
            "command": " ".join(sys.argv),
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            report_path.name: {
                "sha256": _file_sha256(report_path),
                "bytes": report_path.stat().st_size,
            }
        },
    }
    _write_new(args.output / "manifest.json", canonical_json_bytes(manifest))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
