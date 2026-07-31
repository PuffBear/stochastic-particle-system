#!/usr/bin/env python3
"""Execute the frozen SPS-WO-06 timestep-convergence diagnostic.

Validates that dt=0.02 is a converged discretization for the unique-yield
endpoint. Three candidate timestep sizes are tested; the primary gate checks
that the dt=0.02 mean oracle-minus-stationary contrast lies within 2 particles
of the dt=0.01 (finer reference) mean contrast. Seeds are permanently
ineligible for confirmation.
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
from particle_benchmark.runner import (
    EVENT_KEYED_TIE_SCHEME,
    _jsonable,
    _ndarray_sha256,
    _policy_actions,
)


FROZEN_DT_VALUES = (0.01, 0.02, 0.04)
FROZEN_SEEDS = tuple(range(3001, 3009))
FROZEN_ALPHA = 0.06
FROZEN_ALPHAS = (0.0, 0.06)
FROZEN_POLICIES = ("stationary", "full_state_interception_oracle")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 7_031
CONVERGENCE_TOLERANCE = 2.0  # particles; gate: |dt02_mean - dt01_mean| <= tolerance
EXPERIMENT_ID = "SPS-WO-06-TIMESTEP-CONVERGENCE"


def _evaluation_steps(dt: float) -> int:
    """Derive evaluation step count so the window stays fixed at 0.16 / (0.12 * dt)."""
    return math.ceil(0.16 / (0.12 * dt))


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
    seed: int, alpha: float, policy_id: str, dt: float
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
    checksums = _stream_checksums(env)
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

    def outcome(seed: int, policy: str, dt: float, alpha: float = FROZEN_ALPHA) -> float:
        return float(by_key[(seed, alpha, policy, dt)]["unique_team_capture_yield"])

    contrasts_by_dt: dict[float, dict[str, object]] = {}
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

    mean_dt01 = float(contrasts_by_dt[0.01]["mean"])
    mean_dt02 = float(contrasts_by_dt[0.02]["mean"])
    mean_dt04 = float(contrasts_by_dt[0.04]["mean"])

    correctness_passed = all(
        int(row["executed_steps"]) == _evaluation_steps(float(row["dt"]))
        or int(row["unique_team_capture_yield"]) == 256
        for row in summaries
    )
    convergence_fine_to_medium = abs(mean_dt02 - mean_dt01) <= CONVERGENCE_TOLERANCE
    convergence_medium_to_coarse = abs(mean_dt04 - mean_dt02) <= CONVERGENCE_TOLERANCE

    passed = correctness_passed and convergence_fine_to_medium

    if not correctness_passed:
        interpretation = "correctness_failure"
    elif not convergence_fine_to_medium:
        interpretation = "diverged"
    else:
        interpretation = "converged"

    return {
        "work_order_id": "SPS-WO-06",
        "experiment_id": EXPERIMENT_ID,
        "confirmation_eligible": False,
        "diagnostic_seed_firewall": list(FROZEN_SEEDS),
        "endpoint": "unique team particle captures through inclusive step derived from dt",
        "convergence_tolerance_particles": CONVERGENCE_TOLERANCE,
        "dt_values_tested": list(FROZEN_DT_VALUES),
        "contrasts_by_dt": {
            str(dt): contrasts_by_dt[dt] for dt in FROZEN_DT_VALUES
        },
        "dt_means": {
            "dt_0.01": mean_dt01,
            "dt_0.02": mean_dt02,
            "dt_0.04": mean_dt04,
        },
        "pairwise_differences": {
            "abs_dt02_minus_dt01": abs(mean_dt02 - mean_dt01),
            "abs_dt04_minus_dt02": abs(mean_dt04 - mean_dt02),
        },
        "gate_components": {
            "correctness_and_execution": correctness_passed,
            f"dt02_within_{CONVERGENCE_TOLERANCE}_of_dt01": convergence_fine_to_medium,
            f"dt04_within_{CONVERGENCE_TOLERANCE}_of_dt02_informational": convergence_medium_to_coarse,
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

    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    total = len(FROZEN_SEEDS) * len(FROZEN_ALPHAS) * len(FROZEN_POLICIES) * len(FROZEN_DT_VALUES)
    for dt in FROZEN_DT_VALUES:
        for seed in FROZEN_SEEDS:
            for alpha in FROZEN_ALPHAS:
                for policy in FROZEN_POLICIES:
                    summary, captures = rollout(seed, alpha, policy, dt)
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
