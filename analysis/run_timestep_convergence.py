#!/usr/bin/env python3
"""Execute the frozen SPS-WO-06 timestep-convergence diagnostic.

Validates the dt=0.02 unique-yield endpoint against dt/2 and dt/4 while
holding physical duration fixed.  Every level is driven by the same finest
Brownian path: fine increments are summed exactly before being converted back
to the standard-normal representation consumed by the environment.  Seeds are
permanently ineligible for confirmation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Iterable, Mapping

import numpy as np
import yaml

from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.io import canonical_json_bytes, sha256_json
from particle_benchmark.refinement import aggregate_brownian_increments
from particle_benchmark.runner import (
    EVENT_KEYED_TIE_SCHEME,
    _jsonable,
    _ndarray_sha256,
    _policy_actions,
)


FROZEN_DT_VALUES = (0.02, 0.01, 0.005)
BASE_DT = 0.02
BASE_EVALUATION_STEPS = 67
PHYSICAL_DURATION = BASE_DT * BASE_EVALUATION_STEPS
FROZEN_SEEDS = tuple(range(3001, 3009))
FROZEN_ALPHA = 0.06
FROZEN_ALPHAS = (0.0, 0.06)
FROZEN_POLICIES = ("stationary", "full_state_interception_oracle")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 7_031
CONVERGENCE_TOLERANCE = 1.0  # strict gate: |dt02_mean - dt01_mean| < tolerance
MAX_SIGN_CHANGES = 1
EXPERIMENT_ID = "SPS-WO-06-TIMESTEP-CONVERGENCE"


def _evaluation_steps(dt: float) -> int:
    """Return an integer step count for the frozen 1.34-time-unit window."""
    steps = PHYSICAL_DURATION / dt
    rounded = round(steps)
    if not math.isclose(steps, rounded, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("dt must divide the frozen physical duration exactly")
    return int(rounded)


def coupled_noise_bank(seed: int, *, particle_count: int = 256) -> dict[float, np.ndarray]:
    """Generate one finest tensor and aggregate its Brownian increments.

    The returned arrays are standard normals because ``ParticleCollectorEnv``
    applies ``sqrt(dt)`` itself.  Thus, for every coarse step, the physical
    Brownian increment equals the sum of its corresponding finest increments.
    """
    finest_dt = min(FROZEN_DT_VALUES)
    finest_steps = _evaluation_steps(finest_dt)
    # Use the environment's frozen stream construction, not a new ad-hoc RNG.
    probe = ParticleCollectorEnv(
        ParticleEnvConfig(horizon=finest_steps, dt=finest_dt, particle_count=particle_count)
    )
    probe.reset(seed=seed)
    assert probe._noise is not None
    finest_normals = np.asarray(probe._noise, dtype=np.float64)
    finest_increments = finest_normals * math.sqrt(finest_dt)
    bank: dict[float, np.ndarray] = {}
    for dt in FROZEN_DT_VALUES:
        factor = int(round(dt / finest_dt))
        increments = aggregate_brownian_increments(finest_increments, factor)
        bank[dt] = increments / math.sqrt(dt)
        if bank[dt].shape != (_evaluation_steps(dt), particle_count, 2):
            raise RuntimeError("coupled Brownian tensor has an invalid shape")
    return bank


def unique_capture_yield_through_step(
    events: Iterable[Mapping[str, object]], *, cutoff_step: int
) -> int:
    """Count distinct particle IDs captured on or before ``cutoff_step``."""
    if cutoff_step < 1:
        raise ValueError("cutoff_step must be positive")
    return len(
        {
            int(event["particle_id"])
            for event in events
            if int(event["step"]) <= cutoff_step
        }
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshot(root: Path) -> dict[str, object]:
    paths = (
        root / "analysis" / "run_timestep_convergence.py",
        root / "src" / "particle_benchmark" / "environment.py",
        root / "src" / "particle_benchmark" / "policies.py",
        root / "src" / "particle_benchmark" / "runner.py",
        root / "src" / "particle_benchmark" / "refinement.py",
        root / "src" / "particle_benchmark" / "dynamics" / "capture.py",
        root / "src" / "particle_benchmark" / "dynamics" / "particles.py",
    )
    files = {path.relative_to(root).as_posix(): _file_sha256(path) for path in paths}
    return {"files": files, "aggregate_sha256": sha256_json(files)}


def _stream_checksums(env: ParticleCollectorEnv) -> dict[str, str]:
    assert env.particle_positions is not None
    assert env.collector_positions is not None
    assert env._noise is not None
    assert env._episode_field_kwargs is not None
    assert env.scenario_seed is not None
    tie_key = {
        "scheme": EVENT_KEYED_TIE_SCHEME,
        "scenario_seed": env.scenario_seed,
        "event_key": ["scenario_seed", "one_based_step_index", "stable_particle_id"],
        "candidate_order": "sorted_stable_collector_ids",
    }
    initial_state = {
        "particle_positions_sha256": _ndarray_sha256(env.particle_positions),
        "collector_positions_sha256": _ndarray_sha256(env.collector_positions),
    }
    return {
        "initial_state_sha256": sha256_json(initial_state),
        "brownian_sha256": _ndarray_sha256(env._noise),
        "field_nuisance_sha256": sha256_json(_jsonable(env._episode_field_kwargs)),
        "tie_key_provenance_sha256": sha256_json(tie_key),
    }


def rollout(
    seed: int,
    alpha: float,
    policy_id: str,
    dt: float,
    *,
    coupled_noise: np.ndarray,
    finest_noise_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run one frozen policy-condition episode for a given dt value."""
    evaluation_steps = _evaluation_steps(dt)
    config = ParticleEnvConfig(
        horizon=evaluation_steps, signal_strength=alpha, dt=dt
    )
    env = ParticleCollectorEnv(config)
    observations, reset_info = env.reset(seed=seed)
    if reset_info["captured_total"] != 0 or reset_info["tie_scheme"] != EVENT_KEYED_TIE_SCHEME:
        raise RuntimeError("reset or tie provenance violated the frozen task")
    if coupled_noise.shape != (evaluation_steps, config.particle_count, 2):
        raise ValueError("coupled_noise does not match the requested dt")
    # This is the only sanctioned test hook: initialization and field nuisance
    # remain those produced by reset(seed), while the Brownian stream is
    # replaced by its pre-generated coupled level.
    env._noise = np.asarray(coupled_noise, dtype=np.float64).copy()
    checksums = _stream_checksums(env)
    checksums["coupled_finest_brownian_sha256"] = finest_noise_sha256
    path_length = np.zeros(config.collector_count, dtype=np.float64)
    capture_events: list[dict[str, object]] = []
    first_contact_step: int | None = None

    for step in range(evaluation_steps):
        assert env.collector_positions is not None
        before = env.collector_positions.copy()
        policy_config = (
            {"receding_horizon": 2.0}
            if policy_id == "full_state_interception_oracle"
            else {}
        )
        actions = _policy_actions(
            policy_id,
            observations,
            env,
            step=step,
            random_actions=None,
            policy_config=policy_config,
        )
        if actions.shape != (config.collector_count, 2) or np.any(
            np.linalg.norm(actions, axis=1) > 1.0 + 1e-12
        ):
            raise RuntimeError("policy emitted an illegal action")
        observations, _, terminated, truncated, info = env.step(actions)
        path_length += np.linalg.norm(env.collector_positions - before, axis=1)
        for (particle_id, collector_id), contact_fraction in zip(
            info["captures"], info["contact_time_fractions"], strict=True
        ):
            capture_events.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "dt": dt,
                    "policy_id": policy_id,
                    "step": int(info["step"]),
                    "particle_id": int(particle_id),
                    "collector_id": int(collector_id),
                    "contact_time_fraction": float(contact_fraction),
                }
            )
        first_contact_step = env.first_contact_step
        if info["tie_scheme"] != EVENT_KEYED_TIE_SCHEME:
            raise RuntimeError("tie provenance changed during rollout")
        if terminated:
            if info["captured_total"] != config.particle_count:
                raise RuntimeError("premature termination")
            break
        if truncated and info["step"] != evaluation_steps:
            raise RuntimeError("premature truncation")

    assert env.capture_state is not None
    unique_yield = unique_capture_yield_through_step(
        capture_events, cutoff_step=evaluation_steps
    )
    captured_total = int(np.count_nonzero(env.capture_state.owner >= 0))
    if unique_yield != captured_total or unique_yield != len(capture_events):
        raise RuntimeError("capture-yield uniqueness/accounting failure")
    executed_steps = env.step_count
    if captured_total < config.particle_count and executed_steps != evaluation_steps:
        raise RuntimeError("episode stopped before evaluation window after first contact")
    if any(int(event["step"]) > evaluation_steps for event in capture_events):
        raise RuntimeError("post-window capture entered the endpoint")

    summary: dict[str, object] = {
        "seed": seed,
        "alpha": alpha,
        "dt": dt,
        "kappa": alpha / config.collector_max_speed,
        "policy_id": policy_id,
        "evaluation_steps": evaluation_steps,
        "physical_duration": PHYSICAL_DURATION,
        "brownian_aggregation_factor": int(round(dt / min(FROZEN_DT_VALUES))),
        "executed_steps": executed_steps,
        "first_contact_step": first_contact_step,
        "continued_after_first_contact": bool(
            first_contact_step is not None and executed_steps > first_contact_step
        ),
        "unique_team_capture_yield": unique_yield,
        "per_collector_capture_count": np.bincount(
            env.capture_state.owner[env.capture_state.owner >= 0],
            minlength=config.collector_count,
        ).astype(int).tolist(),
        "collector_path_length": path_length.tolist(),
        "total_collector_path_length": float(np.sum(path_length)),
        "stream_checksums": checksums,
        "contact_model": "fixed_piecewise_specular_reflection_exact_v1",
        "confirmation_eligible": False,
    }
    return summary, capture_events


def _descriptive_contrast(values: np.ndarray, *, name: str, n_seeds: int) -> dict[str, object]:
    if values.shape != (n_seeds,):
        raise ValueError(f"contrast must contain exactly {n_seeds} seeds")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    bootstrap_means = np.mean(values[indices], axis=1)
    return {
        "name": name,
        "seed_level_values": values.astype(float).tolist(),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "positive_seed_count": int(np.count_nonzero(values > 0)),
        "descriptive_paired_bootstrap_95_interval": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def evaluate_gate(summaries: list[dict[str, object]]) -> dict[str, object]:
    n_seeds = len(FROZEN_SEEDS)
    by_key = {
        (int(row["seed"]), float(row["alpha"]), str(row["policy_id"]), float(row["dt"])): row
        for row in summaries
    }
    expected = n_seeds * len(FROZEN_ALPHAS) * len(FROZEN_POLICIES) * len(FROZEN_DT_VALUES)
    if len(by_key) != expected:
        raise RuntimeError("missing or duplicate policy-condition episode")

    coupling_and_provenance_verified = True
    for seed in FROZEN_SEEDS:
        rows = [row for row in summaries if int(row["seed"]) == seed]
        initial_hashes = {
            str(dict(row["stream_checksums"])["initial_state_sha256"]) for row in rows
        }
        field_hashes = {
            str(dict(row["stream_checksums"])["field_nuisance_sha256"]) for row in rows
        }
        finest_hashes = {
            str(dict(row["stream_checksums"])["coupled_finest_brownian_sha256"])
            for row in rows
        }
        if len(initial_hashes) != 1 or len(field_hashes) != 1 or len(finest_hashes) != 1:
            coupling_and_provenance_verified = False
        for dt in FROZEN_DT_VALUES:
            brownian_hashes = {
                str(dict(row["stream_checksums"])["brownian_sha256"])
                for row in rows
                if float(row["dt"]) == dt
            }
            if len(brownian_hashes) != 1:
                coupling_and_provenance_verified = False

    def outcome(seed: int, policy: str, dt: float, alpha: float = FROZEN_ALPHA) -> float:
        return float(by_key[(seed, alpha, policy, dt)]["unique_team_capture_yield"])

    contrasts_by_dt: dict[float, dict[str, object]] = {}
    stationary_signal_minus_null_by_dt: dict[float, dict[str, object]] = {}
    for dt in FROZEN_DT_VALUES:
        d = np.asarray(
            [
                outcome(seed, "full_state_interception_oracle", dt)
                - outcome(seed, "stationary", dt)
                for seed in FROZEN_SEEDS
            ]
        )
        contrasts_by_dt[dt] = _descriptive_contrast(
            d,
            name=f"oracle_minus_stationary_at_alpha_{FROZEN_ALPHA}_dt_{dt}",
            n_seeds=n_seeds,
        )
        passive = np.asarray(
            [
                outcome(seed, "stationary", dt, FROZEN_ALPHA)
                - outcome(seed, "stationary", dt, 0.0)
                for seed in FROZEN_SEEDS
            ]
        )
        stationary_signal_minus_null_by_dt[dt] = _descriptive_contrast(
            passive,
            name=f"stationary_signal_minus_null_dt_{dt}",
            n_seeds=n_seeds,
        )

    mean_dt02 = float(contrasts_by_dt[0.02]["mean"])
    mean_dt01 = float(contrasts_by_dt[0.01]["mean"])
    mean_dt005 = float(contrasts_by_dt[0.005]["mean"])

    contrasts_02 = np.asarray(contrasts_by_dt[0.02]["seed_level_values"], dtype=float)
    contrasts_01 = np.asarray(contrasts_by_dt[0.01]["seed_level_values"], dtype=float)
    signs_02 = np.sign(contrasts_02)
    signs_01 = np.sign(contrasts_01)
    sign_change_count = int(np.count_nonzero(signs_02 != signs_01))

    correctness_passed = all(
        int(row["executed_steps"]) == _evaluation_steps(float(row["dt"]))
        or int(row["unique_team_capture_yield"]) == 256
        for row in summaries
    )
    convergence_base_to_half = abs(mean_dt02 - mean_dt01) < CONVERGENCE_TOLERANCE
    direction_stable = sign_change_count <= MAX_SIGN_CHANGES
    finest_level_informational = abs(mean_dt01 - mean_dt005)

    passed = (
        correctness_passed
        and coupling_and_provenance_verified
        and convergence_base_to_half
        and direction_stable
    )

    if not correctness_passed or not coupling_and_provenance_verified:
        interpretation = "correctness_failure"
    elif not convergence_base_to_half or not direction_stable:
        interpretation = "diverged"
    else:
        interpretation = "converged"

    return {
        "work_order_id": "SPS-WO-06",
        "experiment_id": EXPERIMENT_ID,
        "confirmation_eligible": False,
        "diagnostic_seed_firewall": list(FROZEN_SEEDS),
        "endpoint": "unique team particle captures through fixed physical time 1.34",
        "base_endpoint_equivalence": "67 inclusive steps at dt=0.02",
        "convergence_tolerance_particles": CONVERGENCE_TOLERANCE,
        "dt_values_tested": list(FROZEN_DT_VALUES),
        "contrasts_by_dt": {
            str(dt): contrasts_by_dt[dt] for dt in FROZEN_DT_VALUES
        },
        "stationary_signal_minus_null_by_dt": {
            str(dt): stationary_signal_minus_null_by_dt[dt] for dt in FROZEN_DT_VALUES
        },
        "dt_means": {
            "dt_0.01": mean_dt01,
            "dt_0.02": mean_dt02,
            "dt_0.005": mean_dt005,
        },
        "pairwise_differences": {
            "abs_dt02_minus_dt01": abs(mean_dt02 - mean_dt01),
            "abs_dt01_minus_dt005_informational": finest_level_informational,
            "dt02_to_dt01_seed_sign_change_count": sign_change_count,
        },
        "gate_components": {
            "correctness_and_execution": correctness_passed,
            "coupling_and_provenance_verified": coupling_and_provenance_verified,
            "abs_mean_dt02_minus_dt01_strictly_below_1_particle": convergence_base_to_half,
            "dt02_to_dt01_sign_changes_at_most_1_of_8": direction_stable,
        },
        "convergence_gate_passed": passed,
        "interpretation": interpretation,
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
    if frozen["diagnostic_seeds"] != list(FROZEN_SEEDS):
        raise RuntimeError("frozen config diagnostic_seeds differ from SPS-WO-06 constants")
    if frozen["alphas"] != list(FROZEN_ALPHAS):
        raise RuntimeError("frozen config alphas differ from SPS-WO-06 constants")
    if frozen["policies"] != list(FROZEN_POLICIES):
        raise RuntimeError("frozen config policies differ from SPS-WO-06 constants")
    if frozen["dt_values"] != list(FROZEN_DT_VALUES):
        raise RuntimeError("frozen config dt_values differ from SPS-WO-06 constants")
    if frozen["base_evaluation_steps"] != BASE_EVALUATION_STEPS:
        raise RuntimeError("frozen config base_evaluation_steps differ from SPS-WO-06")
    if float(frozen["physical_duration"]) != PHYSICAL_DURATION:
        raise RuntimeError("frozen config physical_duration differs from SPS-WO-06")
    primary_gate = frozen.get("primary_gate", {})
    if primary_gate.get("mean_absolute_difference_strictly_below_particles") != CONVERGENCE_TOLERANCE:
        raise RuntimeError("frozen config convergence tolerance differs from SPS-WO-06")
    if primary_gate.get("maximum_seed_sign_changes") != MAX_SIGN_CHANGES:
        raise RuntimeError("frozen config sign-change limit differs from SPS-WO-06")
    if frozen.get("coupling") != "finest_standard_normals_then_sum_brownian_increments":
        raise RuntimeError("frozen config does not require finest-level Brownian coupling")

    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    total = len(FROZEN_SEEDS) * len(FROZEN_ALPHAS) * len(FROZEN_POLICIES) * len(FROZEN_DT_VALUES)
    for seed in FROZEN_SEEDS:
        noise_bank = coupled_noise_bank(seed)
        finest_noise_sha256 = _ndarray_sha256(noise_bank[min(FROZEN_DT_VALUES)])
        for dt in FROZEN_DT_VALUES:
            for alpha in FROZEN_ALPHAS:
                for policy in FROZEN_POLICIES:
                    summary, captures = rollout(
                        seed,
                        alpha,
                        policy,
                        dt,
                        coupled_noise=noise_bank[dt],
                        finest_noise_sha256=finest_noise_sha256,
                    )
                    summaries.append(summary)
                    events.extend(captures)
                    print(
                        f"completed {len(summaries)}/{total} "
                        f"dt={dt} seed={seed} alpha={alpha} policy={policy}",
                        file=sys.stderr,
                        flush=True,
                    )
    gate = evaluate_gate(summaries)
    runtime_seconds = time.perf_counter() - started

    args.output.mkdir(parents=False)
    summaries_path = args.output / "episode_summaries.jsonl"
    events_path = args.output / "capture_events.jsonl"
    gate_path = args.output / "convergence_report.json"
    _write_new(summaries_path, b"".join(canonical_json_bytes(row) for row in summaries))
    _write_new(events_path, b"".join(canonical_json_bytes(row) for row in events))
    _write_new(gate_path, canonical_json_bytes(gate))
    root = Path(__file__).resolve().parents[1]
    command = " ".join(sys.argv)
    artifacts = {
        path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
        for path in (summaries_path, events_path, gate_path)
    }
    manifest = {
        "work_order_id": "SPS-WO-06",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_eligible": False,
        "repository": {
            "full_name": "PuffBear/stochastic-particle-system",
            "branch": "research-autonomy",
            "base_commit_sha": args.repository_base_commit,
            "workspace_changes_included": True,
        },
        "source_snapshot": _source_snapshot(root),
        "frozen_config": {
            "path": args.config.as_posix(),
            "sha256": _file_sha256(args.config),
            "contents": frozen,
        },
        "runtime": {
            "command": command,
            "seconds": runtime_seconds,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "episode_count": len(summaries),
        "environment_step_count": int(sum(int(row["executed_steps"]) for row in summaries)),
        "artifacts": artifacts,
    }
    manifest_path = args.output / "manifest.json"
    _write_new(manifest_path, canonical_json_bytes(manifest))
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
