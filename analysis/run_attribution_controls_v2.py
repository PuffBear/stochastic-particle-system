#!/usr/bin/env python3
"""Execute SPS-WO-07B attribution control runs for the improved shared_summary_v2 policy.

Tests whether the count-weighted + field/density blend controller
(capacity_matched_velocity_controller_v2, shared=True) outperforms the
capacity-matched independent baseline on fresh diagnostic seeds 5001-5008.

The gate mirrors SPS-WO-07 except that:
  - policy under test is shared_summary_v2 (not shared_summary)
  - seeds are 5001-5008 (never used in any prior work order)
  - upstream requires that SPS-WO-06 convergence gate has passed

These seeds are permanently ineligible for confirmation.
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


FROZEN_SEEDS = tuple(range(5001, 5009))
FROZEN_ALPHA = 0.06
FROZEN_ALPHAS = (0.0, 0.06)
EVALUATION_STEPS = 67
FROZEN_DT = 0.02
FROZEN_POLICIES = (
    "capacity_matched_independent",
    "shared_summary_v2",
    "stationary",
    "full_state_interception_oracle",
)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 73_031
EXPERIMENT_ID = "SPS-WO-07B-ATTRIBUTION-V2"


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
        root / "analysis" / "run_attribution_controls_v2.py",
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


def rollout(seed: int, alpha: float, policy_id: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run one frozen 67-step policy-condition episode without first-contact stop."""
    config = ParticleEnvConfig(
        horizon=EVALUATION_STEPS, signal_strength=alpha, dt=FROZEN_DT
    )
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

    matched_streams_verified = True
    for seed in FROZEN_SEEDS:
        baseline: dict[str, str] | None = None
        for alpha in FROZEN_ALPHAS:
            for policy in FROZEN_POLICIES:
                checks = dict(by_key[(seed, alpha, policy)]["stream_checksums"])
                if baseline is None:
                    baseline = checks
                elif checks != baseline:
                    matched_streams_verified = False

    def outcome(seed: int, policy: str, alpha: float = FROZEN_ALPHA) -> float:
        return float(by_key[(seed, alpha, policy)]["unique_team_capture_yield"])

    shared_v2_minus_independent = np.asarray(
        [
            outcome(seed, "shared_summary_v2")
            - outcome(seed, "capacity_matched_independent")
            for seed in FROZEN_SEEDS
        ]
    )
    oracle_minus_shared_v2 = np.asarray(
        [
            outcome(seed, "full_state_interception_oracle")
            - outcome(seed, "shared_summary_v2")
            for seed in FROZEN_SEEDS
        ]
    )
    shared_v2_minus_stationary = np.asarray(
        [
            outcome(seed, "shared_summary_v2")
            - outcome(seed, "stationary")
            for seed in FROZEN_SEEDS
        ]
    )
    oracle_minus_independent = np.asarray(
        [
            outcome(seed, "full_state_interception_oracle")
            - outcome(seed, "capacity_matched_independent")
            for seed in FROZEN_SEEDS
        ]
    )

    primary_contrast = _descriptive_contrast(
        shared_v2_minus_independent,
        name="shared_v2_minus_independent_at_alpha_0.06",
    )
    attr_check_1 = _descriptive_contrast(
        oracle_minus_shared_v2,
        name="oracle_minus_shared_v2_at_alpha_0.06",
    )
    attr_check_2 = _descriptive_contrast(
        shared_v2_minus_stationary,
        name="shared_v2_minus_stationary_at_alpha_0.06",
    )
    reference_contrast = _descriptive_contrast(
        oracle_minus_independent,
        name="oracle_minus_independent_at_alpha_0.06",
    )

    correctness_passed = all(
        int(row["executed_steps"]) == EVALUATION_STEPS
        or int(row["unique_team_capture_yield"]) == 256
        for row in summaries
    )
    coordination_direction_passed = float(primary_contrast["mean"]) > 0.0
    positive_seed_count_passed = int(primary_contrast["positive_seed_count"]) >= 5
    shared_v2_mean = float(np.mean([outcome(seed, "shared_summary_v2") for seed in FROZEN_SEEDS]))
    independent_mean = float(
        np.mean([outcome(seed, "capacity_matched_independent") for seed in FROZEN_SEEDS])
    )
    stationary_mean = float(np.mean([outcome(seed, "stationary") for seed in FROZEN_SEEDS]))
    shared_v2_exceeds_stationary = shared_v2_mean > stationary_mean
    passive_adjusted_ordering = (
        float(np.mean([outcome(seed, "shared_summary_v2") - outcome(seed, "stationary") for seed in FROZEN_SEEDS]))
        > float(np.mean([outcome(seed, "capacity_matched_independent") - outcome(seed, "stationary") for seed in FROZEN_SEEDS]))
    )
    passed = all(
        (
            correctness_passed,
            matched_streams_verified,
            coordination_direction_passed,
            positive_seed_count_passed,
            shared_v2_exceeds_stationary,
            passive_adjusted_ordering,
        )
    )

    if not correctness_passed or not matched_streams_verified:
        interpretation = "invalid_correctness_or_provenance_failure"
    elif not coordination_direction_passed or not positive_seed_count_passed:
        interpretation = "sharing_v2_does_not_help"
    elif not shared_v2_exceeds_stationary or not passive_adjusted_ordering:
        interpretation = "sharing_v2_effect_fails_attribution_controls"
    else:
        interpretation = "sharing_v2_helps_confirmation_warranted"

    return {
        "work_order_id": "SPS-WO-07B",
        "experiment_id": EXPERIMENT_ID,
        "confirmation_eligible": False,
        "diagnostic_seed_firewall": list(FROZEN_SEEDS),
        "endpoint": "unique team particle captures through inclusive step 67",
        "primary_contrast": primary_contrast,
        "attribution_check_1_oracle_minus_shared_v2": attr_check_1,
        "attribution_check_2_shared_v2_minus_stationary": attr_check_2,
        "reference_oracle_minus_independent": reference_contrast,
        "arm_means": {
            "shared_summary_v2": shared_v2_mean,
            "capacity_matched_independent": independent_mean,
            "stationary": stationary_mean,
        },
        "null_alpha_yields": [
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
            "matched_streams_verified": matched_streams_verified,
            "mean_shared_v2_minus_independent_positive": coordination_direction_passed,
            "positive_shared_v2_minus_independent_in_at_least_5_of_8": positive_seed_count_passed,
            "mean_shared_v2_exceeds_mean_stationary": shared_v2_exceeds_stationary,
            "passive_adjusted_ordering": passive_adjusted_ordering,
        },
        "attribution_gate_passed": passed,
        "full_confirmation_run_warranted": passed,
        "interpretation": interpretation,
        "claim_effect": (
            "SPS-C03 confirmation run is authorized if gate passes; "
            "this diagnostic is not itself confirmatory"
        ),
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
    parser.add_argument(
        "--upstream-gate",
        type=Path,
        required=True,
        help="immutable SPS-WO-06 convergence_report.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    if len(args.repository_base_commit) != 40:
        raise ValueError("repository base commit must be a full 40-character SHA")
    frozen = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if frozen["diagnostic_seeds"] != list(FROZEN_SEEDS):
        raise RuntimeError("frozen config diagnostic_seeds differ from SPS-WO-07B constants")
    if frozen["alphas"] != list(FROZEN_ALPHAS):
        raise RuntimeError("frozen config alphas differ from SPS-WO-07B constants")
    if frozen["policies"] != list(FROZEN_POLICIES):
        raise RuntimeError("frozen config policies differ from SPS-WO-07B constants")
    if frozen["evaluation_steps"] != EVALUATION_STEPS:
        raise RuntimeError("frozen config evaluation_steps differ from SPS-WO-07B constants")
    if float(frozen["dt"]) != FROZEN_DT:
        raise RuntimeError("frozen config dt differs from SPS-WO-07B constants")
    if frozen["bootstrap_draws"] != BOOTSTRAP_DRAWS:
        raise RuntimeError("frozen config bootstrap_draws differ from SPS-WO-07B constants")
    if frozen["bootstrap_seed"] != BOOTSTRAP_SEED:
        raise RuntimeError("frozen config bootstrap_seed differs from SPS-WO-07B constants")
    if frozen.get("upstream_work_order") != "SPS-WO-06":
        raise RuntimeError("frozen config must require SPS-WO-06")
    upstream = json.loads(args.upstream_gate.read_text(encoding="utf-8"))
    if upstream.get("work_order_id") != "SPS-WO-06" or not upstream.get(
        "convergence_gate_passed", False
    ):
        raise RuntimeError("SPS-WO-07B is blocked until a valid SPS-WO-06 gate passes")

    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    total = len(FROZEN_SEEDS) * len(FROZEN_ALPHAS) * len(FROZEN_POLICIES)
    for seed in FROZEN_SEEDS:
        for alpha in FROZEN_ALPHAS:
            for policy in FROZEN_POLICIES:
                summary, captures = rollout(seed, alpha, policy)
                summaries.append(summary)
                events.extend(captures)
                print(
                    f"completed {len(summaries)}/{total} seed={seed} alpha={alpha} policy={policy}",
                    file=sys.stderr,
                    flush=True,
                )
    gate = evaluate_gate(summaries)
    runtime_seconds = time.perf_counter() - started

    args.output.mkdir(parents=False)
    summaries_path = args.output / "episode_summaries.jsonl"
    events_path = args.output / "capture_events.jsonl"
    gate_path = args.output / "attribution_gate.json"
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
        "work_order_id": "SPS-WO-07B",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_eligible": False,
        "repository": {
            "full_name": "PuffBear/stochastic-particle-system",
            "branch": "claude/research-autonomy-review-rwjyjg",
            "base_commit_sha": args.repository_base_commit,
            "workspace_changes_included": True,
        },
        "source_snapshot": _source_snapshot(root),
        "frozen_config": {
            "path": args.config.as_posix(),
            "sha256": _file_sha256(args.config),
            "contents": frozen,
        },
        "upstream_gate": {
            "path": args.upstream_gate.as_posix(),
            "sha256": _file_sha256(args.upstream_gate),
            "work_order_id": upstream["work_order_id"],
            "convergence_gate_passed": True,
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
