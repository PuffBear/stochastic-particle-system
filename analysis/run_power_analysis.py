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

Implementation note:
  Power is estimated via Monte Carlo simulation.  Each trial draws ``n_seeds``
  iid observations from N(effect, sd^2) and computes a one-sided studentized
  bootstrap lower bound directly using numpy — this avoids the per-call
  overhead of ``simultaneous_paired_lower_bounds`` in a Python loop while
  producing numerically identical results for the single-grid-point case.
  The bootstrap resampling is vectorized in batches over trials for speed.

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
SIMULATION_TRIALS = 2_000
SIMULATION_RNG_SEED = 8_421
TARGET_POWER = 0.80
EXPERIMENT_ID = "SPS-POWER-ANALYSIS-COORDINATION"

# Bootstrap batch size: controls peak memory per batch
# (BATCH_SIZE × BOOTSTRAP_DRAWS × max_n_seeds × 8 bytes)
# 20 × 10_000 × 64 × 8 ≈ 102 MB — safe on standard hardware
_BOOTSTRAP_BATCH_SIZE = 20


def _simulate_power(
    n_seeds: int,
    assumed_sd: float,
    effect: float,
    *,
    simulation_trials: int,
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> float:
    """Estimate power via vectorized Monte Carlo simulation.

    For each of ``simulation_trials`` synthetic datasets, draws ``n_seeds``
    iid observations from N(effect, assumed_sd^2), then computes the one-sided
    studentized bootstrap lower confidence bound for the mean using direct
    numpy vectorization batched over trials.  Power is the fraction of trials
    in which the lower bound exceeds zero.

    The bootstrap resampling is batched over trials (see ``_BOOTSTRAP_BATCH_SIZE``)
    to avoid allocating the full (simulation_trials × bootstrap_draws × n_seeds)
    array at once.
    """
    # Draw all trial data at once: shape (simulation_trials, n_seeds)
    samples = rng.normal(loc=effect, scale=assumed_sd, size=(simulation_trials, n_seeds))
    obs_means = np.mean(samples, axis=1)  # (simulation_trials,)
    obs_ses = np.std(samples, axis=1, ddof=1) / np.sqrt(n_seeds)  # (simulation_trials,)

    successes = 0
    batch = _BOOTSTRAP_BATCH_SIZE
    for start in range(0, simulation_trials, batch):
        end = min(start + batch, simulation_trials)
        b = end - start
        batch_samples = samples[start:end]      # (b, n_seeds)
        batch_obs_means = obs_means[start:end]  # (b,)
        batch_obs_ses = obs_ses[start:end]       # (b,)

        # Boot indices: (b, bootstrap_draws, n_seeds)
        boot_idx = rng.integers(0, n_seeds, size=(b, bootstrap_draws, n_seeds))
        # Gather bootstrap samples via advanced indexing
        # batch_samples[trial_index, boot_idx[trial_index, draw_index, :]]
        b_idx = np.arange(b)[:, None, None]  # (b, 1, 1) for broadcasting
        boot_samples = batch_samples[b_idx, boot_idx]  # (b, bootstrap_draws, n_seeds)

        boot_means = np.mean(boot_samples, axis=2)                              # (b, bootstrap_draws)
        boot_ses = np.std(boot_samples, axis=2, ddof=1) / np.sqrt(n_seeds)     # (b, bootstrap_draws)

        # Studentized shortfall: (observed_mean - boot_mean) / boot_se per draw
        shortfall = np.where(
            boot_ses > 1e-15,
            (batch_obs_means[:, None] - boot_means) / boot_ses,
            0.0,
        )  # (b, bootstrap_draws)

        # Critical value: 95th percentile of shortfall across bootstrap draws
        critical = np.quantile(shortfall, 0.95, axis=1, method="higher")  # (b,)
        lcbs = batch_obs_means - critical * batch_obs_ses                  # (b,)
        successes += int(np.sum(lcbs > 0))

    return successes / simulation_trials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulation-based power analysis for SPS-C03 coordination diagnostic"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sd_values = np.linspace(ASSUMED_SD_RANGE[0], ASSUMED_SD_RANGE[1], ASSUMED_SD_STEPS)
    rng = np.random.default_rng(SIMULATION_RNG_SEED)

    power_table: list[dict[str, object]] = []
    print(
        f"{'n_seeds':>8}  {'assumed_sd':>10}  {'power':>8}",
        file=sys.stderr,
        flush=True,
    )
    for n_seeds in SEED_COUNTS:
        for assumed_sd in sd_values:
            power = _simulate_power(
                n_seeds,
                float(assumed_sd),
                COORDINATION_MINIMUM_EFFECT,
                simulation_trials=SIMULATION_TRIALS,
                bootstrap_draws=BOOTSTRAP_DRAWS,
                rng=rng,
            )
            row = {
                "n_seeds": n_seeds,
                "assumed_sd": float(assumed_sd),
                "effect": COORDINATION_MINIMUM_EFFECT,
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
        "simulation_trials": SIMULATION_TRIALS,
        "simulation_rng_seed": SIMULATION_RNG_SEED,
        "target_power": TARGET_POWER,
        "test": "one-sided studentized bootstrap lower bound > 0 at 95% confidence",
        "bootstrap_implementation": "direct numpy vectorization (batched over trials)",
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
