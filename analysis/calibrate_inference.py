"""Deterministic Monte Carlo calibration for the frozen simultaneous LCB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from particle_benchmark.metrics.inference import simultaneous_paired_lower_bounds


def _wilson_upper(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2)
    )
    return float((centre + radius) / denominator)


def _calibrate_model(
    *, name: str, trials: int, bootstrap_draws: int, rng: np.random.Generator
) -> dict[str, object]:
    false_crossings = 0
    for trial in range(trials):
        if name == "correlated_continuous_null":
            shared = rng.normal(size=(12, 1))
            null_effects = 0.65 * shared + np.sqrt(1.0 - 0.65**2) * rng.normal(size=(12, 5))
        elif name == "zero_inflated_symmetric_discrete_null":
            active = rng.random(size=(12, 5)) < 0.35
            signs = rng.choice([-1.0, 1.0], size=(12, 5))
            magnitudes = rng.choice([0.025, 0.05, 0.10], size=(12, 5))
            null_effects = active * signs * magnitudes
        else:
            raise ValueError(name)
        bounds = simultaneous_paired_lower_bounds(
            null_effects,
            bootstrap_draws=bootstrap_draws,
            bootstrap_seed=90_000 + trial,
        )
        false_crossings += bool(np.any(bounds.lower > 0.0))
    rate = false_crossings / trials
    upper = _wilson_upper(false_crossings, trials)
    return {
        "model": name,
        "false_crossings": false_crossings,
        "empirical_familywise_false_crossing_rate": rate,
        "wilson_95pct_upper": upper,
        "passed": upper <= 0.07,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(2047)
    models = [
        _calibrate_model(
            name=name,
            trials=args.trials,
            bootstrap_draws=args.bootstrap_draws,
            rng=rng,
        )
        for name in (
            "correlated_continuous_null",
            "zero_inflated_symmetric_discrete_null",
        )
    ]
    result = {
        "calibration_models": models,
        "scenario_seeds_per_grid_point": 12,
        "nonzero_grid_points": 5,
        "trials": args.trials,
        "bootstrap_draws_per_trial": args.bootstrap_draws,
        "simulation_seed": 2047,
        "bootstrap_seed_rule": "90000 + trial_index",
        "acceptance_rule": "Wilson 95% upper bound <= 0.07 for each model",
        "passed": all(bool(model["passed"]) for model in models),
        "interpretation": "Synthetic calibration only; not evidence for SPS-C01.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
