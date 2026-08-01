"""Run the bounded SPS-P01 calibration pilot and preserve compact raw summaries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from particle_benchmark.environment import ParticleEnvConfig
from particle_benchmark.io import PreparedArtifact, canonical_json_bytes, write_immutable_artifacts
from particle_benchmark.metrics.inference import (
    grid_censored_boundary,
    persistent_grid_censored_boundary,
    simultaneous_paired_lower_bounds,
)
from particle_benchmark.runner import (
    PairExperimentConfig,
    RepositoryProvenance,
    run_matched_pair,
)


RHO_GRID = np.array([0.0, 0.10, 0.25, 0.50, 1.00, 2.00])
SEEDS = tuple(range(1001, 1013))


def _json_artifact(filename: str, value: object) -> PreparedArtifact:
    return PreparedArtifact(filename, canonical_json_bytes(value), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--experiment-id", default="SPS-P01")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"pilot output already exists: {args.output}")

    repository = RepositoryProvenance(
        full_name="PuffBear/stochastic-particle-system",
        branch="research-autonomy",
        commit_sha=args.commit_sha,
        dirty=True,
    )
    base_environment = ParticleEnvConfig(
        arena_size=(1.0, 1.0),
        dt=0.02,
        horizon=400,
        particle_count=256,
        diffusion_sigma=0.06,
        collector_count=4,
        collector_max_speed=0.12,
        sensing_radius=0.16,
        collector_radius=0.012,
        particle_radius=0.0,
        field_family="uniform",
        signal_strength=0.0,
        field_kwargs={},
        capture_geometry="fixed",
        nearest_particles_k=32,
        include_particle_velocity=True,
        include_teammates=True,
    )
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    raw_rows: list[dict[str, object]] = []

    conditions: list[tuple[str, str, int, float]] = []
    for rho in RHO_GRID:
        conditions.append(("primary_four_independent", "local_flow_v1", 4, float(rho)))
    for policy_id in (
        "stationary",
        "pregenerated_random",
        "coverage",
        "density_greedy",
        "privileged_upstream_oracle",
    ):
        conditions.append(("strongest_signal_control", policy_id, 4, 2.0))
    conditions.append(("single_collector_descriptive", "local_flow_v1", 1, 2.0))

    with tempfile.TemporaryDirectory(prefix="sps-p01-pairs-") as temporary:
        for treatment, policy_id, collector_count, rho in conditions:
            alpha = float(rho * base_environment.diffusion_sigma / math.sqrt(base_environment.dt))
            environment = ParticleEnvConfig(
                **{
                    **asdict(base_environment),
                    "collector_count": collector_count,
                    "signal_strength": alpha,
                }
            )
            for seed in SEEDS:
                config = PairExperimentConfig(
                    experiment_id=args.experiment_id,
                    scenario_seed=seed,
                    signal_strength=alpha,
                    policy_id=policy_id,
                    environment=environment,
                    repository=repository,
                    created_at_utc=created_at,
                    config_path="configs/experiments/pilot.yaml",
                    run_command="PYTHONPATH=src python3 analysis/run_scripted_pilot.py",
                    stop_after_first_interception=True,
                )
                pair_dir = Path(temporary) / f"{treatment}-{policy_id}-m{collector_count}-rho{rho}-seed{seed}"
                result = run_matched_pair(config, pair_dir)
                raw_rows.append(
                    {
                        "treatment": treatment,
                        "policy_id": policy_id,
                        "collector_count": collector_count,
                        "rho": rho,
                        "signal_strength": alpha,
                        **result.summary,
                    }
                )

    primary_rows = [
        row
        for row in raw_rows
        if row["treatment"] == "primary_four_independent" and float(row["rho"]) > 0.0
    ]
    primary_matrix = np.array(
        [
            [
                next(
                    float(row["matched_first_interception_gain"])
                    for row in primary_rows
                    if int(row["scenario_seed"]) == seed and float(row["rho"]) == rho
                )
                for rho in RHO_GRID[1:]
            ]
            for seed in SEEDS
        ]
    )
    bounds = simultaneous_paired_lower_bounds(
        primary_matrix,
        confidence=0.95,
        bootstrap_draws=10_000,
        bootstrap_seed=73_031,
    )
    halfwidth = bounds.mean - bounds.lower
    first_boundary = grid_censored_boundary(RHO_GRID[1:], bounds.lower)
    persistent_boundary = persistent_grid_censored_boundary(RHO_GRID[1:], bounds.lower)
    primary_analysis = {
        "status": "exploratory_calibration_only",
        "rho_grid": RHO_GRID[1:].tolist(),
        "scenario_seeds": list(SEEDS),
        "mean_gain": bounds.mean.tolist(),
        "simultaneous_one_sided_95pct_lcb": bounds.lower.tolist(),
        "halfwidth": halfwidth.tolist(),
        "bootstrap_draws": bounds.bootstrap_draws,
        "bootstrap_seed": 73_031,
        "studentized_max_critical_value": bounds.critical_value,
        "first_positive_lcb_boundary": first_boundary,
        "persistent_positive_lcb_boundary": persistent_boundary,
        "sample_covariance": np.cov(primary_matrix, rowvar=False, ddof=1).tolist(),
        "paired_standard_error": (
            np.std(primary_matrix, axis=0, ddof=1) / math.sqrt(len(SEEDS))
        ).tolist(),
        "interpretation": "Pilot seeds calibrate precision only and cannot update SPS-C01.",
    }
    maximum_halfwidth = float(np.max(halfwidth))
    raw_requirement = len(SEEDS) * (maximum_halfwidth / 0.05) ** 2
    planned = int(4 * math.ceil(max(24.0, 1.25 * raw_requirement) / 4.0))
    seed_budget = {
        "target_simultaneous_halfwidth": 0.05,
        "pilot_maximum_halfwidth": maximum_halfwidth,
        "raw_requirement": raw_requirement,
        "inflation": 1.25,
        "minimum": 24,
        "maximum_authorized": 64,
        "planned_evidence_seeds": planned,
        "within_authorized_cap": planned <= 64,
        "evidence_seed_range": [2001, 2000 + planned] if planned <= 64 else None,
    }

    strongest = [row for row in raw_rows if float(row["rho"]) == 2.0]
    baseline_summary: dict[str, object] = {}
    for treatment, policy_id, collector_count, _ in conditions:
        key = f"{treatment}:{policy_id}:M{collector_count}"
        if key in baseline_summary:
            continue
        values = np.array(
            [
                float(row["matched_first_interception_gain"])
                for row in strongest
                if row["treatment"] == treatment
                and row["policy_id"] == policy_id
                and int(row["collector_count"]) == collector_count
            ]
        )
        if values.size:
            baseline_summary[key] = {
                "n": int(values.size),
                "mean_gain": float(np.mean(values)),
                "median_gain": float(np.median(values)),
                "censored_signal_fraction": float(
                    np.mean(
                        [
                            bool(row["signal"]["censored"])
                            for row in strongest
                            if row["treatment"] == treatment
                            and row["policy_id"] == policy_id
                            and int(row["collector_count"]) == collector_count
                        ]
                    )
                ),
            }

    reflection_guards = int(
        sum(
            int(row[condition]["guarded_reflection_pairs"])
            for row in raw_rows
            for condition in ("null", "signal")
        )
    )
    confirmatory_blockers = ["pilot seeds are calibration-only"]
    if reflection_guards:
        confirmatory_blockers.append(
            "exact segmented reflected-path contact was not active for every pair"
        )
    confirmatory_blockers.append(
        "primary policy is four independent replicas and fails the AAMAS multi-agent gate"
    )
    quality = {
        "pair_count": len(raw_rows),
        "zero_signal_pairs": sum(float(row["rho"]) == 0.0 for row in raw_rows),
        "all_zero_signal_identity_checks_passed": all(
            bool(row["zero_signal_identity_checked"])
            for row in raw_rows
            if float(row["rho"]) == 0.0
        ),
        "guarded_reflection_pairs": reflection_guards,
        "reflected_path_contact_gate_passed": reflection_guards == 0,
        "confirmatory_interpretation_authorized": False,
        "confirmatory_blockers": confirmatory_blockers,
        "runtime_seconds": time.perf_counter() - started,
    }
    payload = b"".join(canonical_json_bytes(row) for row in raw_rows)
    artifacts = [
        PreparedArtifact("pair_summaries.jsonl", payload, "application/x-ndjson"),
        _json_artifact("primary_analysis.json", primary_analysis),
        _json_artifact("baseline_summary.json", baseline_summary),
        _json_artifact("seed_budget.json", seed_budget),
        _json_artifact("quality_gates.json", quality),
    ]
    write_immutable_artifacts(args.output, artifacts)
    print(json.dumps({"primary": primary_analysis, "baselines": baseline_summary, "seed_budget": seed_budget, "quality": quality}, sort_keys=True))


if __name__ == "__main__":
    main()
