#!/usr/bin/env python3
"""Analyze FR-B3 factorial results without optional statistics dependencies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import yaml

from particle_benchmark.io import canonical_json_bytes


SHARED = "shared_summary_v2"
INDEPENDENT = "capacity_matched_independent"


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("no FR-B3 episode summaries found")
    return rows


def paired_differences(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, dict[int, float]]:
    outcomes: dict[tuple[str, int, str], float] = {}
    for row in rows:
        if row.get("study") != "factorial":
            raise ValueError("factorial analyzer received a non-factorial row")
        key = (str(row["condition_id"]), int(row["seed"]), str(row["policy_id"]))
        if key in outcomes:
            raise ValueError(f"duplicate episode summary: {key}")
        outcomes[key] = float(row["unique_team_capture_yield"])
    conditions = sorted({key[0] for key in outcomes})
    seeds = sorted({key[1] for key in outcomes})
    result: dict[str, dict[int, float]] = {}
    for condition in conditions:
        result[condition] = {}
        for seed in seeds:
            shared_key = (condition, seed, SHARED)
            independent_key = (condition, seed, INDEPENDENT)
            if shared_key not in outcomes or independent_key not in outcomes:
                raise ValueError(f"missing primary paired arm for {(condition, seed)}")
            result[condition][seed] = outcomes[shared_key] - outcomes[independent_key]
    return result


def condition_coordinates(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, tuple[float, float, float]]:
    coordinates: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        groups = dict(row["dimensionless_groups"])
        value = (float(groups["rho"]), float(groups["kappa"]), float(groups["eta"]))
        condition = str(row["condition_id"])
        if condition in coordinates:
            np.testing.assert_allclose(coordinates[condition], value, rtol=1e-12, atol=1e-12)
        else:
            coordinates[condition] = value
    return coordinates


def _features(
    coordinate: tuple[float, float, float],
    anchor: tuple[float, float, float],
    *,
    include_eta: bool,
) -> np.ndarray:
    x, y, z = (
        math.log2(value / reference)
        for value, reference in zip(coordinate, anchor, strict=True)
    )
    if include_eta:
        return np.asarray([1.0, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z])
    return np.asarray([1.0, x, y, x * x, y * y, x * y])


def leave_one_cell_out_rmse(
    coordinates: list[tuple[float, float, float]],
    outcomes: np.ndarray,
    *,
    anchor: tuple[float, float, float],
    include_eta: bool,
) -> tuple[float, list[float]]:
    predictions: list[float] = []
    for held_out in range(len(coordinates)):
        train = [index for index in range(len(coordinates)) if index != held_out]
        design = np.vstack(
            [_features(coordinates[index], anchor, include_eta=include_eta) for index in train]
        )
        coefficients = np.linalg.lstsq(design, outcomes[train], rcond=None)[0]
        prediction = float(
            _features(coordinates[held_out], anchor, include_eta=include_eta) @ coefficients
        )
        predictions.append(prediction)
    errors = np.asarray(predictions) - outcomes
    return float(np.sqrt(np.mean(errors * errors))), predictions


def _model_comparison(
    differences: dict[str, dict[int, float]],
    coordinates_by_id: dict[str, tuple[float, float, float]],
    *,
    anchor: tuple[float, float, float],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    condition_ids = sorted(differences)
    seeds = sorted(next(iter(differences.values())))
    if any(sorted(values) != seeds for values in differences.values()):
        raise ValueError("all factorial cells must use the same seed panel")
    matrix = np.asarray(
        [[differences[condition][seed] for seed in seeds] for condition in condition_ids],
        dtype=np.float64,
    )
    coordinates = [coordinates_by_id[condition] for condition in condition_ids]
    means = np.mean(matrix, axis=1)
    rmse_two, predictions_two = leave_one_cell_out_rmse(
        coordinates, means, anchor=anchor, include_eta=False
    )
    rmse_three, predictions_three = leave_one_cell_out_rmse(
        coordinates, means, anchor=anchor, include_eta=True
    )

    rng = np.random.default_rng(bootstrap_seed)
    ratios = np.empty(bootstrap_draws, dtype=np.float64)
    improvements = np.empty(bootstrap_draws, dtype=np.float64)
    for draw in range(bootstrap_draws):
        sampled = rng.integers(0, len(seeds), size=len(seeds))
        draw_means = np.mean(matrix[:, sampled], axis=1)
        draw_two, _ = leave_one_cell_out_rmse(
            coordinates, draw_means, anchor=anchor, include_eta=False
        )
        draw_three, _ = leave_one_cell_out_rmse(
            coordinates, draw_means, anchor=anchor, include_eta=True
        )
        ratios[draw] = draw_three / draw_two if draw_two > 0 else math.inf
        improvements[draw] = draw_two - draw_three
    ratio = rmse_three / rmse_two if rmse_two > 0 else math.inf
    return {
        "condition_ids": condition_ids,
        "cell_mean_differences": means.tolist(),
        "two_axis_loco_predictions": predictions_two,
        "three_axis_loco_predictions": predictions_three,
        "two_axis_loco_rmse": rmse_two,
        "three_axis_loco_rmse": rmse_three,
        "three_to_two_axis_rmse_ratio": ratio,
        "bootstrap_ratio_95_interval": [
            float(np.quantile(ratios, 0.025)),
            float(np.quantile(ratios, 0.975)),
        ],
        "bootstrap_rmse_improvement_90_interval": [
            float(np.quantile(improvements, 0.05)),
            float(np.quantile(improvements, 0.95)),
        ],
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
    }


def _cell_summaries(
    differences: dict[str, dict[int, float]],
    coordinates: dict[str, tuple[float, float, float]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for condition in sorted(differences):
        values = np.asarray(list(differences[condition].values()), dtype=np.float64)
        summaries.append(
            {
                "condition_id": condition,
                "rho": coordinates[condition][0],
                "kappa": coordinates[condition][1],
                "eta": coordinates[condition][2],
                "n": len(values),
                "mean_paired_gain": float(np.mean(values)),
                "sample_standard_deviation": float(np.std(values, ddof=1)),
                "positive_seed_count": int(np.count_nonzero(values > 0)),
                "zero_seed_count": int(np.count_nonzero(values == 0)),
            }
        )
    return summaries


def analyze(
    rows: list[dict[str, object]], protocol: Mapping[str, object]
) -> dict[str, object]:
    differences = paired_differences(rows)
    coordinates = condition_coordinates(rows)
    expected_seed_panel = sorted(int(seed) for seed in protocol["seeds"])
    observed_seed_panel = sorted(next(iter(differences.values())))
    if observed_seed_panel != expected_seed_panel:
        raise ValueError("factorial output does not contain the frozen seed panel")
    expected_cell_count = math.prod(
        len(tuple(values)) for values in dict(protocol["factorial_axes"]).values()
    )
    if len(differences) != expected_cell_count:
        raise ValueError("factorial output does not contain every frozen cell")
    anchor_config = dict(protocol["anchor"])
    anchor = tuple(float(anchor_config[key]) for key in ("rho", "kappa", "eta"))
    analysis_config = dict(protocol["analysis"])
    comparison = _model_comparison(
        differences,
        coordinates,
        anchor=anchor,
        bootstrap_draws=int(analysis_config["bootstrap_draws"]),
        bootstrap_seed=int(analysis_config["bootstrap_seed"]),
    )
    threshold = float(analysis_config["two_axis_rejection_rmse_ratio"])
    improvement_quantile = float(
        analysis_config["bootstrap_improvement_lower_quantile"]
    )
    if improvement_quantile != 0.05:
        raise ValueError("FR-B3 currently supports the frozen 0.05 lower quantile")
    ratio = float(comparison["three_to_two_axis_rmse_ratio"])
    improvement_lower = float(
        comparison["bootstrap_rmse_improvement_90_interval"][0]
    )
    practically_meaningful = ratio <= threshold
    statistically_supported = improvement_lower > 0.0
    rejected = practically_meaningful and statistically_supported
    return {
        "experiment_id": protocol["experiment_id"],
        "analysis_status": "designed_confirmatory_analysis",
        "primary_estimand": "E[Y(shared_summary_v2)-Y(capacity_matched_independent)]",
        "cell_summaries": _cell_summaries(differences, coordinates),
        "predictive_model_comparison": comparison,
        "decision_rule": {
            "observed_rmse_ratio_must_not_exceed": threshold,
            "bootstrap_rmse_improvement_lower_quantile": improvement_quantile,
            "bootstrap_rmse_improvement_lower_bound_must_exceed": 0.0,
            "practically_meaningful_improvement": practically_meaningful,
            "statistically_supported_improvement": statistically_supported,
            "two_axis_rejected": rejected,
            "interpretation": (
                "eta materially improves held-out prediction"
                if rejected
                else (
                    "the result does not meet both the predictive effect-size "
                    "and uncertainty gates"
                )
            ),
        },
        "warning": "A non-rejection is not proof that rho and kappa are sufficient.",
    }


def validate_rescaling_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("study") != "rescaling_audit":
            raise ValueError("rescaling validator received a non-audit row")
        grouped.setdefault((int(row["seed"]), str(row["policy_id"])), []).append(row)
    failures: list[dict[str, object]] = []
    for (seed, policy), group in sorted(grouped.items()):
        yields = {int(row["unique_team_capture_yield"]) for row in group}
        state_hashes = {str(row["normalized_final_state_sha256"]) for row in group}
        if len(yields) != 1 or len(state_hashes) != 1:
            failures.append(
                {
                    "seed": seed,
                    "policy_id": policy,
                    "yield_values": sorted(yields),
                    "normalized_state_hash_count": len(state_hashes),
                }
            )
    return {
        "audit_passed": not failures,
        "comparison_count": len(grouped),
        "failures": failures,
        "criterion": "identical yield and normalized final-state checksum across all rescalings",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=("factorial", "rescaling-audit"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    rows = load_rows(args.summaries)
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = (
        analyze(rows, protocol)
        if args.study == "factorial"
        else validate_rescaling_audit(rows)
    )
    with args.output.open("xb") as handle:
        handle.write(canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
