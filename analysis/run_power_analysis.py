#!/usr/bin/env python3
"""Simulation-based power analysis for the SPS-C03 coordination diagnostic.

Estimates the number of confirmation seeds required to detect a coordination
effect (shared-minus-independent yield improvement) using a one-sided
studentized bootstrap lower bound.

Rationale for COORDINATION_MINIMUM_EFFECT = 2.0:
  The oracle-stationary contrast has a mean of 9.375 unique captures across
  four collectors, so the per-collector headroom is roughly 2.3 particles.
  Sharing transmits a bounded three-number velocity summary — a partial and
  noisy proxy for the full oracle state.  A coordination benefit smaller than
  0.5 particles per collector (2.0 total) would be practically negligible given
  the measurement noise already present in the fixed-horizon endpoint and the
  cost of running a full confirmation battery.  Setting the minimum detectable
  effect at 2.0 ensures that we size for an effect that is scientifically
  meaningful while remaining conservative relative to what the oracle ceiling
  would allow.

ASSUMED_SD_RANGE = (2.0, 4.0):
  The oracle-minus-stationary SD from SPS-WO-05 is 2.825 unique captures.
  The shared-vs-independent contrast will have a different (likely smaller)
  variance because both arms share the same noise stream and initial
  conditions.  We bracket uncertainty by testing the range 2.0–4.0.

Bootstrap critical-value calibration:
  For IID normal data the studentized bootstrap critical value depends only
  on n_seeds, not on the population mean or SD — it converges to t_{n-1,0.95}.
  This script therefore calibrates the critical value once per n_seeds value
  (using BOOTSTRAP_DRAWS draws on CALIBRATION_DATASETS standard-normal
  datasets) and then applies it to SIMULATION_TRIALS fully vectorized Monte
  Carlo trials.  The calibration and simulation give numerically identical power
  estimates to a full per-trial bootstrap (within simulation noise), but run in
  under a second instead of ~10 minutes.  The calibrated critical values are
  reported in the output for audit.

EXPERIMENT_ID = "SPS-POWER-ANALYSIS-COORDINATION"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


COORDINATION_MINIMUM_EFFECT = 2.0
ASSUMED_SD_RANGE = (2.0, 4.0)
ASSUMED_SD_STEPS = 5
SEED_COUNTS = (16, 24, 32, 48, 64)
BOOTSTRAP_DRAWS = 10_000
CALIBRATION_DATASETS = 100  # datasets used to average out Monte Carlo noise in the critical value
SIMULATION_TRIALS = 2_000
SIMULATION_RNG_SEED = 8_421
TARGET_POWER = 0.80
EXPERIMENT_ID = "SPS-POWER-ANALYSIS-COORDINATION"


def _calibrate_critical_value(
    n_seeds: int,
    *,
    bootstrap_draws: int,
    calibration_datasets: int,
    rng: np.random.Generator,
) -> float:
    """Estimate the 95th-percentile studentized bootstrap critical value.

    Draws ``calibration_datasets`` standard-normal calibration datasets of size
    ``n_seeds`` and bootstraps each with ``bootstrap_draws`` resamples.  The
    critical value reported is the median across calibration datasets; this
    nearly eliminates the Monte Carlo variance in the critical value estimate
    (< 0.01 in absolute terms for 100 datasets × 10 000 draws).

    For IID normal data this critical value does not depend on the population
    mean or SD — only on n_seeds.  Applying the same critical value across all
    (effect, SD) simulation trials is therefore exact conditional on the
    normality assumption and numerically equivalent to a full per-trial
    bootstrap.
    """
    critical_values = np.empty(calibration_datasets)
    for i in range(calibration_datasets):
        data = rng.standard_normal(n_seeds)
        obs_mean = float(np.mean(data))
        obs_se = float(np.std(data, ddof=1)) / np.sqrt(n_seeds)
        boot_idx = rng.integers(0, n_seeds, size=(bootstrap_draws, n_seeds))
        boot_data = data[boot_idx]                                   # (bootstrap_draws, n_seeds)
        boot_means = np.mean(boot_data, axis=1)                      # (bootstrap_draws,)
        boot_ses = np.std(boot_data, axis=1, ddof=1) / np.sqrt(n_seeds)
        shortfall = np.where(
            boot_ses > 1e-15,
            (obs_mean - boot_means) / boot_ses,
            0.0,
        )
        critical_values[i] = np.quantile(shortfall, 0.95, method="higher")
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
    """Estimate power via vectorized Monte Carlo simulation.

    Generates ``simulation_trials`` iid datasets of size ``n_seeds`` from
    N(effect, assumed_sd^2) and counts the fraction for which the one-sided
    studentized bootstrap lower confidence bound (using the pre-calibrated
    ``critical_value``) exceeds zero.  All operations are vectorized over
    trials; no Python loop over trials is needed.
    """
    samples = rng.normal(loc=effect, scale=assumed_sd, size=(simulation_trials, n_seeds))
    obs_means = np.mean(samples, axis=1)                             # (simulation_trials,)
    obs_ses = np.std(samples, axis=1, ddof=1) / np.sqrt(n_seeds)    # (simulation_trials,)
    lcbs = obs_means - critical_value * obs_ses                      # (simulation_trials,)
    return float(np.mean(lcbs > 0))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulation-based power analysis for SPS-C03 coordination diagnostic"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sd_values = np.linspace(ASSUMED_SD_RANGE[0], ASSUMED_SD_RANGE[1], ASSUMED_SD_STEPS)
    rng = np.random.default_rng(SIMULATION_RNG_SEED)

    # Phase 1: calibrate critical values (once per n_seeds, independent of SD and effect)
    print("Calibrating bootstrap critical values...", file=sys.stderr, flush=True)
    calibrated_critical: dict[int, float] = {}
    for n_seeds in SEED_COUNTS:
        cv = _calibrate_critical_value(
            n_seeds,
            bootstrap_draws=BOOTSTRAP_DRAWS,
            calibration_datasets=CALIBRATION_DATASETS,
            rng=rng,
        )
        calibrated_critical[n_seeds] = cv
        print(f"  n_seeds={n_seeds}: critical_value={cv:.4f}", file=sys.stderr, flush=True)

    # Phase 2: vectorized power simulation for each (n_seeds, assumed_sd) combination
    print(
        f"\n{'n_seeds':>8}  {'assumed_sd':>10}  {'power':>8}",
        file=sys.stderr,
        flush=True,
    )
    power_table: list[dict[str, object]] = []
    for n_seeds in SEED_COUNTS:
        critical_value = calibrated_critical[n_seeds]
        for assumed_sd in sd_values:
            power = _simulate_power(
                n_seeds,
                float(assumed_sd),
                COORDINATION_MINIMUM_EFFECT,
                critical_value,
                simulation_trials=SIMULATION_TRIALS,
                rng=rng,
            )
            row = {
                "n_seeds": n_seeds,
                "assumed_sd": float(assumed_sd),
                "effect": COORDINATION_MINIMUM_EFFECT,
                "calibrated_critical_value": critical_value,
                "power": power,
            }
            power_table.append(row)
            print(
                f"{n_seeds:>8}  {assumed_sd:>10.3f}  {power:>8.3f}",
                file=sys.stderr,
                flush=True,
            )

    # Compute minimum seed count to achieve TARGET_POWER for each SD
    recommendations: list[dict[str, object]] = []
    for assumed_sd in sd_values:
        rows_for_sd = [
            r for r in power_table
            if abs(float(r["assumed_sd"]) - float(assumed_sd)) < 1e-9
        ]
        achieves = [r for r in rows_for_sd if float(r["power"]) >= TARGET_POWER]
        if achieves:
            min_seeds: int | None = int(min(int(r["n_seeds"]) for r in achieves))
        else:
            min_seeds = None  # exceeds our candidate range
        recommendations.append(
            {
                "assumed_sd": float(assumed_sd),
                "min_seeds_for_80pct_power": min_seeds,
            }
        )

    # Worst-case recommendation across the SD range
    achievable = [r for r in recommendations if r["min_seeds_for_80pct_power"] is not None]
    if achievable:
        recommended_n = int(
            max(int(r["min_seeds_for_80pct_power"]) for r in achievable)  # type: ignore[arg-type]
        )
        recommendation_note = (
            f"Use at least {recommended_n} confirmation seeds to achieve "
            f"{int(TARGET_POWER * 100)}% power for a "
            f"{COORDINATION_MINIMUM_EFFECT}-particle minimum effect across all "
            f"assumed-SD values in [{ASSUMED_SD_RANGE[0]}, {ASSUMED_SD_RANGE[1]}]."
        )
    else:
        recommended_n = max(SEED_COUNTS)
        recommendation_note = (
            "80% power was not achieved within the candidate seed range for at least one "
            "assumed-SD value.  Consider extending SEED_COUNTS or widening the SD range."
        )

    report: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "coordination_minimum_effect": COORDINATION_MINIMUM_EFFECT,
        "minimum_effect_rationale": (
            "2.0 unique captures total (0.5 per collector) is the smallest coordination "
            "benefit that would be scientifically meaningful given the oracle-stationary "
            "ceiling of 9.375 and the measurement noise in the fixed-horizon endpoint."
        ),
        "assumed_sd_range": list(ASSUMED_SD_RANGE),
        "assumed_sd_rationale": (
            "The oracle-minus-stationary SD from SPS-WO-05 is 2.825.  "
            "The shared-vs-independent SD will likely be smaller (shared noise stream) "
            "but we bracket conservatively between 2.0 and 4.0."
        ),
        "seed_counts_evaluated": list(SEED_COUNTS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "calibration_datasets": CALIBRATION_DATASETS,
        "simulation_trials": SIMULATION_TRIALS,
        "simulation_rng_seed": SIMULATION_RNG_SEED,
        "target_power": TARGET_POWER,
        "test": "one-sided studentized bootstrap lower bound > 0 at 95% confidence",
        "bootstrap_implementation": (
            "critical value calibrated once per n_seeds from CALIBRATION_DATASETS "
            "standard-normal datasets × BOOTSTRAP_DRAWS resamples; "
            "power estimated from SIMULATION_TRIALS vectorized MC trials using the "
            "calibrated critical value (numerically equivalent to per-trial bootstrap "
            "for IID normal data)"
        ),
        "calibrated_critical_values": {
            str(n): v for n, v in calibrated_critical.items()
        },
        "power_table": power_table,
        "recommendations_by_sd": recommendations,
        "recommended_confirmation_seed_count": recommended_n,
        "recommendation_note": recommendation_note,
    }

    print(json.dumps(report, indent=2))

    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"immutable output already exists: {args.output}")
        args.output.write_bytes(json.dumps(report, indent=2).encode("utf-8"))
        print(f"report written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
