#!/usr/bin/env python3
"""Execute the frozen SPS-WO-05 fixed-horizon capture-yield diagnostic.

This runner deliberately changes only the endpoint and first-contact stopping
rule. It uses the existing environment and policies unchanged, records all
unique capture events through step 67, and writes a new immutable artifact
directory. The diagnostic seeds are permanently ineligible for confirmation.
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


FROZEN_SEEDS = tuple(range(2001, 2009))
FROZEN_ALPHAS = (0.0, 0.06)
FROZEN_POLICIES = (
    "stationary",
    "true_field_upstream_control",
    "full_state_interception_oracle",
)
EVALUATION_STEPS = 67
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 6_705
EXPERIMENT_ID = "SPS-WO-05-YIELD-GATE-R1"


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
        root / "analysis" / "run_yield_gate.py",
        root / "src" / "particle_benchmark" / "environment.py",
        root / "src" / "particle_benchmark" / "policies.py",
        root / "src" / "particle_benchmark" / "runner.py",
        root / "src" / "particle_benchmark" / "dynamics" / "capture.py",
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


def rollout(seed: int, alpha: float, policy_id: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run one frozen 67-step policy-condition episode without first-contact stop."""
    config = ParticleEnvConfig(horizon=EVALUATION_STEPS, signal_strength=alpha)
    env = ParticleCollectorEnv(config)
    observations, reset_info = env.reset(seed=seed)
    if reset_info["captured_total"] != 0 or reset_info["tie_scheme"] != EVENT_KEYED_TIE_SCHEME:
        raise RuntimeError("reset or tie provenance violated the frozen task")
    checksums = _stream_checksums(env)
    path_length = np.zeros(config.collector_count, dtype=np.float64)
    capture_events: list[dict[str, object]] = []
    first_contact_step: int | None = None

    for step in range(EVALUATION_STEPS):
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
        if truncated and info["step"] != EVALUATION_STEPS:
            raise RuntimeError("premature truncation")

    assert env.capture_state is not None
    unique_yield = unique_capture_yield_through_step(
        capture_events, cutoff_step=EVALUATION_STEPS
    )
    captured_total = int(np.count_nonzero(env.capture_state.owner >= 0))
    if unique_yield != captured_total or unique_yield != len(capture_events):
        raise RuntimeError("capture-yield uniqueness/accounting failure")
    executed_steps = env.step_count
    if captured_total < config.particle_count and executed_steps != EVALUATION_STEPS:
        raise RuntimeError("episode stopped before step 67 after first contact")
    if any(int(event["step"]) > EVALUATION_STEPS for event in capture_events):
        raise RuntimeError("post-window capture entered the endpoint")

    summary: dict[str, object] = {
        "seed": seed,
        "alpha": alpha,
        "kappa": alpha / config.collector_max_speed,
        "policy_id": policy_id,
        "evaluation_steps": EVALUATION_STEPS,
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


def _descriptive_contrast(values: np.ndarray, *, name: str) -> dict[str, object]:
    if values.shape != (len(FROZEN_SEEDS),):
        raise ValueError("contrast must contain exactly eight frozen seeds")
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
    by_key = {
        (int(row["seed"]), float(row["alpha"]), str(row["policy_id"])): row
        for row in summaries
    }
    expected = len(FROZEN_SEEDS) * len(FROZEN_ALPHAS) * len(FROZEN_POLICIES)
    if len(by_key) != expected:
        raise RuntimeError("missing or duplicate policy-condition episode")
    for seed in FROZEN_SEEDS:
        baseline: dict[str, str] | None = None
        for alpha in FROZEN_ALPHAS:
            for policy in FROZEN_POLICIES:
                checks = dict(by_key[(seed, alpha, policy)]["stream_checksums"])
                if baseline is None:
                    baseline = checks
                elif checks != baseline:
                    raise RuntimeError(f"matched stream checksum failure for seed {seed}")

    def outcome(seed: int, policy: str, alpha: float = 0.06) -> float:
        return float(by_key[(seed, alpha, policy)]["unique_team_capture_yield"])

    d = np.asarray(
        [outcome(seed, "full_state_interception_oracle") - outcome(seed, "stationary") for seed in FROZEN_SEEDS]
    )
    u = np.asarray(
        [outcome(seed, "full_state_interception_oracle") - outcome(seed, "true_field_upstream_control") for seed in FROZEN_SEEDS]
    )
    primary = _descriptive_contrast(d, name="oracle_minus_stationary_at_alpha_0.06")
    targeting = _descriptive_contrast(u, name="oracle_minus_true_field_at_alpha_0.06")
    correctness_passed = all(
        int(row["executed_steps"]) == EVALUATION_STEPS
        or int(row["unique_team_capture_yield"]) == 256
        for row in summaries
    )
    primary_passed = float(primary["mean"]) >= 4.0
    direction_passed = int(primary["positive_seed_count"]) >= 6
    targeting_passed = float(targeting["mean"]) > 0.0
    passed = correctness_passed and primary_passed and direction_passed and targeting_passed
    if float(primary["mean"]) <= 0.0 or float(targeting["mean"]) <= 0.0:
        interpretation = "reversed"
    elif not passed:
        interpretation = "null"
    else:
        interpretation = "positive"
    return {
        "work_order_id": "SPS-WO-05",
        "experiment_id": EXPERIMENT_ID,
        "confirmation_eligible": False,
        "diagnostic_seed_firewall": list(FROZEN_SEEDS),
        "endpoint": "unique team particle captures through inclusive step 67",
        "primary_contrast": primary,
        "targeting_contrast": targeting,
        "null_yields": [
            {
                "seed": seed,
                "policy_id": policy,
                "unique_team_capture_yield": int(outcome(seed, policy, 0.0)),
            }
            for seed in FROZEN_SEEDS
            for policy in FROZEN_POLICIES
        ],
        "gate_components": {
            "correctness_and_execution": correctness_passed,
            "mean_oracle_minus_stationary_at_least_4": primary_passed,
            "positive_in_at_least_6_of_8_seeds": direction_passed,
            "mean_oracle_minus_true_field_positive": targeting_passed,
            "artifact_and_matched_stream_checks": True,
        },
        "oracle_gate_passed": passed,
        "downstream_sharing_run_authorized": passed,
        "interpretation": interpretation,
        "claim_effect": "SPS-C03 remains blocked; this diagnostic cannot support a coordination claim",
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
    if frozen["diagnostic_seeds"] != list(FROZEN_SEEDS) or frozen["alpha"] != list(FROZEN_ALPHAS):
        raise RuntimeError("frozen config differs from SPS-WO-05 constants")
    if frozen["policies"] != list(FROZEN_POLICIES) or frozen["evaluation_steps"] != EVALUATION_STEPS:
        raise RuntimeError("frozen config differs from SPS-WO-05 constants")

    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    for seed in FROZEN_SEEDS:
        for alpha in FROZEN_ALPHAS:
            for policy in FROZEN_POLICIES:
                summary, captures = rollout(seed, alpha, policy)
                summaries.append(summary)
                events.extend(captures)
                print(
                    f"completed {len(summaries)}/48 seed={seed} alpha={alpha} policy={policy}",
                    file=sys.stderr,
                    flush=True,
                )
    gate = evaluate_gate(summaries)
    runtime_seconds = time.perf_counter() - started

    args.output.mkdir(parents=False)
    summaries_path = args.output / "episode_summaries.jsonl"
    events_path = args.output / "capture_events.jsonl"
    gate_path = args.output / "oracle_gate.json"
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
        "work_order_id": "SPS-WO-05",
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
