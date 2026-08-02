#!/usr/bin/env python3
"""Simulation calibration for the FR-B3 seed budget and decision rule."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from particle_benchmark.io import canonical_json_bytes


HISTORICAL_PAIRED_SD = 2.442070776496733
ANCHOR_GAIN = 1.1875
RMSE_RATIO_THRESHOLD = 0.80
IMPROVEMENT_LOWER_QUANTILE = 0.05
POWER_TARGET = 0.80
RNG_SEED = 81_427


def _coordinates() -> np.ndarray:
    """Return log2 coordinates for the frozen 3 x 3 x 3 design."""

    return np.asarray(
        [
            (rho, kappa, eta)
            for rho in (-1.0, 0.0, 1.0)
            for kappa in (-1.0, 0.0, 1.0)
            for eta in (-1.0, 0.0, 1.0)
        ],
        dtype=np.float64,
    )


def _design_matrices() -> tuple[np.ndarray, np.ndarray]:
    coordinates = _coordinates()
    x, y, z = coordinates.T
    two_axis = np.column_stack(
        [np.ones(27), x, y, x * x, y * y, x * y]
    )
    three_axis = np.column_stack(
        [np.ones(27), x, y, z, x * x, y * y, z * z, x * y, x * z, y * z]
    )
    return two_axis, three_axis


def _loco_operator(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pseudoinverse = np.linalg.pinv(design)
    leverage = np.diag(design @ pseudoinverse)
    return pseudoinverse, 1.0 - leverage


def loco_rmses(outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two- and three-axis LOCO RMSE for one or many outcome vectors.

    ``outcomes`` has shape ``(27, draws)``. The PRESS residual identity makes
    the simulation exactly equivalent to refitting each leave-one-cell-out
    model while avoiding thousands of repeated least-squares calls.
    """

    if outcomes.ndim != 2 or outcomes.shape[0] != 27:
        raise ValueError("outcomes must have shape (27, draws)")
    results: list[np.ndarray] = []
    for design in _design_matrices():
        pseudoinverse, denominator = _loco_operator(design)
        residual = outcomes - design @ (pseudoinverse @ outcomes)
        press_residual = residual / denominator[:, None]
        results.append(np.sqrt(np.mean(press_residual * press_residual, axis=0)))
    return results[0], results[1]


def _mean_surface(high_low_eta_effect: float) -> np.ndarray:
    x, y, z = _coordinates().T
    two_axis_surface = ANCHOR_GAIN + 0.25 * x - 0.20 * y + 0.10 * x * y
    return two_axis_surface + 0.5 * high_low_eta_effect * z


def simulate_scenario(
    *,
    seed_count: int,
    cross_cell_correlation: float,
    high_low_eta_effect: float,
    trials: int,
    bootstrap_draws: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    if seed_count < 2 or trials <= 0 or bootstrap_draws <= 0:
        raise ValueError("seed_count, trials, and bootstrap_draws must be positive")
    if not 0.0 <= cross_cell_correlation < 1.0:
        raise ValueError("cross_cell_correlation must be in [0, 1)")
    mean = _mean_surface(high_low_eta_effect)
    common_sd = HISTORICAL_PAIRED_SD * math.sqrt(cross_cell_correlation)
    residual_sd = HISTORICAL_PAIRED_SD * math.sqrt(
        1.0 - cross_cell_correlation
    )
    decisions = np.zeros(trials, dtype=np.bool_)
    observed_ratios = np.empty(trials, dtype=np.float64)

    for trial in range(trials):
        common_seed_effect = rng.normal(0.0, common_sd, size=(1, seed_count))
        cell_noise = rng.normal(0.0, residual_sd, size=(27, seed_count))
        seed_level_outcomes = mean[:, None] + common_seed_effect + cell_noise
        two_rmse, three_rmse = loco_rmses(
            np.mean(seed_level_outcomes, axis=1, keepdims=True)
        )
        observed_ratio = float(three_rmse[0] / two_rmse[0])
        observed_ratios[trial] = observed_ratio

        indices = rng.integers(
            0, seed_count, size=(bootstrap_draws, seed_count)
        )
        bootstrap_means = np.mean(seed_level_outcomes[:, indices], axis=2)
        bootstrap_two, bootstrap_three = loco_rmses(bootstrap_means)
        improvement_lower = float(
            np.quantile(
                bootstrap_two - bootstrap_three,
                IMPROVEMENT_LOWER_QUANTILE,
            )
        )
        decisions[trial] = (
            observed_ratio <= RMSE_RATIO_THRESHOLD and improvement_lower > 0.0
        )

    return {
        "seed_count": seed_count,
        "cross_cell_correlation": cross_cell_correlation,
        "high_low_eta_effect_captures": high_low_eta_effect,
        "decision_rate": float(np.mean(decisions)),
        "median_observed_rmse_ratio": float(np.median(observed_ratios)),
        "observed_rmse_ratio_10_90_interval": [
            float(np.quantile(observed_ratios, 0.10)),
            float(np.quantile(observed_ratios, 0.90)),
        ],
    }


def calibrate(
    *,
    seed_counts: tuple[int, ...],
    correlations: tuple[float, ...],
    effects: tuple[float, ...],
    trials: int,
    bootstrap_draws: int,
    rng_seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(rng_seed)
    scenarios = [
        simulate_scenario(
            seed_count=seed_count,
            cross_cell_correlation=correlation,
            high_low_eta_effect=effect,
            trials=trials,
            bootstrap_draws=bootstrap_draws,
            rng=rng,
        )
        for seed_count in seed_counts
        for correlation in correlations
        for effect in effects
    ]
    conservative_target = next(
        row
        for row in scenarios
        if row["seed_count"] == 64
        and row["cross_cell_correlation"] == 0.0
        and row["high_low_eta_effect_captures"] == 1.0
    )
    return {
        "experiment_id": "FR-B3-CATCHABILITY-DESIGN-CALIBRATION",
        "simulation_model": (
            "broad linear eta effect across all nine rho-kappa slices with "
            "Gaussian seed-level paired contrasts"
        ),
        "historical_paired_sd": HISTORICAL_PAIRED_SD,
        "anchor_gain": ANCHOR_GAIN,
        "decision_rule": {
            "observed_three_to_two_axis_rmse_ratio_max": RMSE_RATIO_THRESHOLD,
            "bootstrap_rmse_improvement_lower_quantile": (
                IMPROVEMENT_LOWER_QUANTILE
            ),
            "bootstrap_rmse_improvement_lower_bound_min": 0.0,
        },
        "minimum_target_high_low_eta_effect_captures": 1.0,
        "power_target": POWER_TARGET,
        "trials_per_scenario": trials,
        "bootstrap_draws_per_trial": bootstrap_draws,
        "rng_seed": rng_seed,
        "scenarios": scenarios,
        "recommended_seed_count": 64,
        "recommendation_passed": bool(
            float(conservative_target["decision_rate"]) >= POWER_TARGET
        ),
        "limitations": [
            "Calibration uses the paired SD from the single SPS-C03 anchor.",
            "Power applies to a broad linear eta effect, not a sparse one-cell effect.",
            "Gaussian simulation does not reproduce discrete or heavy-tailed yields.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = calibrate(
        seed_counts=(32, 48, 64),
        correlations=(0.0, 0.25, 0.50),
        effects=(0.0, 1.0, 1.5),
        trials=args.trials,
        bootstrap_draws=args.bootstrap_draws,
        rng_seed=args.rng_seed,
    )
    payload = canonical_json_bytes(report)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"immutable output already exists: {args.output}")
        with args.output.open("xb") as handle:
            handle.write(payload)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
