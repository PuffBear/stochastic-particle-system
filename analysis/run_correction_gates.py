#!/usr/bin/env python3
"""Bounded SPS-P03 catchable-grid mechanism diagnostics.

Diagnostic seeds are never confirmation eligible.  Outputs are written once
to a new directory; existing evidence is never replaced.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from particle_benchmark.dynamics.fields import field_velocity
from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.runner import _policy_actions


SEEDS = tuple(range(1001, 1009))
ALPHAS = (0.0, 0.03, 0.06, 0.12)
POLICIES = (
    "stationary",
    "true_field_upstream_control",
    "full_state_interception_oracle",
    "local_flow_v1",
)


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denom <= 1e-15 else float(np.dot(left, right) / denom)


def _angular_error(left: np.ndarray, right: np.ndarray) -> float | None:
    cosine = _cosine(left, right)
    return None if cosine is None else float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rollout(
    seed: int, alpha: float, policy_id: str, *, particle_count: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config = ParticleEnvConfig(signal_strength=alpha, particle_count=particle_count)
    env = ParticleCollectorEnv(config)
    observations, _ = env.reset(seed=seed)
    path_length = np.zeros(config.collector_count)
    step_rows: list[dict[str, object]] = []
    for step in range(config.horizon):
        before = np.asarray(env.collector_positions).copy()
        actions = _policy_actions(
            policy_id, observations, env, step=step, random_actions=None,
            policy_config={"receding_horizon": 2.0} if policy_id == "full_state_interception_oracle" else {},
        )
        assert env._episode_field_kwargs is not None
        latent = field_velocity(before, config.field_family, alpha, **env._episode_field_kwargs)
        next_observations, _, terminated, truncated, info = env.step(actions)
        path_length += np.linalg.norm(np.asarray(env.collector_positions) - before, axis=1)
        for agent_id, observation in enumerate(observations):
            present = np.asarray(observation["particle_mask"], dtype=bool)
            valid = present & np.asarray(observation["velocity_valid_mask"], dtype=bool)
            slots = np.asarray(observation["particles"], dtype=float)
            estimated = np.mean(slots[valid, 2:4], axis=0) if np.any(valid) else np.zeros(2)
            nearest = slots[np.flatnonzero(present)[0], :2] if np.any(present) else np.zeros(2)
            wall_distance = float(np.min(np.r_[before[agent_id], np.asarray(config.arena_size) - before[agent_id]]))
            step_rows.append({
                "seed": seed, "alpha": alpha, "kappa": alpha / config.collector_max_speed,
                "policy_id": policy_id, "step": step + 1, "agent_id": agent_id,
                "visible_count": int(np.count_nonzero(present)),
                "valid_track_count": int(np.count_nonzero(valid)),
                "nonzero_action": bool(np.linalg.norm(actions[agent_id]) > 1e-12),
                "estimated_velocity": estimated.tolist(), "true_field_velocity": latent[agent_id].tolist(),
                "velocity_magnitude_error": float(np.linalg.norm(estimated - latent[agent_id])),
                "velocity_angular_error_radians": _angular_error(estimated, latent[agent_id]),
                "action_estimated_alignment": _cosine(actions[agent_id], estimated),
                "action_true_field_alignment": _cosine(actions[agent_id], latent[agent_id]),
                "action_nearest_particle_alignment": _cosine(actions[agent_id], nearest),
                "wall_distance": wall_distance, "wall_proximity": wall_distance <= config.sensing_radius,
                "path_length_cumulative": float(path_length[agent_id]),
                "capture_count": int(sum(cid == agent_id for _, cid in info["captures"])),
            })
        observations = next_observations
        if env.first_contact_step is not None or terminated or truncated:
            break
    first = env.first_contact_step or config.horizon + 1
    owner = None
    if env.capture_state is not None and env.first_contact_step is not None:
        captured = np.flatnonzero(env.capture_state.owner >= 0)
        owner = int(env.capture_state.owner[captured[0]]) if captured.size else None
    summary = {
        "seed": seed, "alpha": alpha, "kappa": alpha / config.collector_max_speed,
        "policy_id": policy_id, "first_interception_step": first,
        "restricted_first_interception_time": min(first, config.horizon) * config.dt,
        "censored": env.first_contact_step is None, "first_contact_owner": owner,
        "path_length": path_length.tolist(),
        "swept_area_proxy": float(np.sum(path_length) * 2.0 * config.collector_radius),
        "fraction_steps_with_valid_track": float(np.mean([r["valid_track_count"] > 0 for r in step_rows])),
        "fraction_steps_with_nonzero_action": float(np.mean([r["nonzero_action"] for r in step_rows])),
    }
    return summary, step_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--particle-count", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    started = time.perf_counter()
    summaries: list[dict[str, object]] = []
    steps: list[dict[str, object]] = []
    for seed in SEEDS:
        for alpha in ALPHAS:
            for policy_id in POLICIES:
                summary, rows = rollout(
                    seed, alpha, policy_id, particle_count=args.particle_count
                )
                summaries.append(summary)
                steps.extend(rows)

    by_key = {(r["seed"], r["alpha"], r["policy_id"]): r for r in summaries}
    gate_points = []
    for alpha in ALPHAS[1:]:
        directions = 0
        adjusted = []
        absolute_oracle_minus_stationary = []
        absolute_oracle_minus_true_field = []
        for seed in SEEDS:
            def first(policy: str, strength: float) -> float:
                return float(by_key[(seed, strength, policy)]["first_interception_step"])
            oracle_gain = (first("full_state_interception_oracle", 0.0) - first("full_state_interception_oracle", alpha)) / 400.0
            stationary_gain = (first("stationary", 0.0) - first("stationary", alpha)) / 400.0
            value = oracle_gain - stationary_gain
            adjusted.append(value)
            directions += value > 0
            absolute_oracle_minus_stationary.append(first("stationary", alpha) - first("full_state_interception_oracle", alpha))
            absolute_oracle_minus_true_field.append(first("true_field_upstream_control", alpha) - first("full_state_interception_oracle", alpha))
        gate_points.append({
            "alpha": alpha, "kappa": alpha / 0.12,
            "mean_passive_adjusted_gain": float(np.mean(adjusted)),
            "positive_seed_count": int(directions),
            "mean_steps_earlier_than_stationary": float(np.mean(absolute_oracle_minus_stationary)),
            "mean_steps_earlier_than_true_field_control": float(np.mean(absolute_oracle_minus_true_field)),
            "passes": bool(np.mean(adjusted) > 0 and directions >= 6 and np.mean(absolute_oracle_minus_stationary) > 0 and np.mean(absolute_oracle_minus_true_field) > 0),
        })
    gate_passed = any(point["passes"] for point in gate_points)
    args.output.mkdir(parents=True)
    summary_bytes = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in summaries)
    step_bytes = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in steps)
    (args.output / "episode_summaries.jsonl").write_bytes(summary_bytes)
    (args.output / "collector_step_diagnostics.jsonl").write_bytes(step_bytes)
    analysis = {
        "experiment_id": args.experiment_id, "confirmation_eligible": False,
        "oracle_gate_passed": gate_passed, "gate_points": gate_points,
        "downstream_sharing_run_authorized": gate_passed,
        "interpretation": "Diagnostic feasibility only; failure stops sharing execution.",
    }
    (args.output / "oracle_gate.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    manifest = {
        "experiment_id": args.experiment_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"PYTHONPATH=src python analysis/run_correction_gates.py --output {args.output} --experiment-id {args.experiment_id} --particle-count {args.particle_count}",
        "python": platform.python_version(), "numpy": np.__version__, "seeds": list(SEEDS),
        "particle_count": args.particle_count,
        "alphas": list(ALPHAS), "policies": list(POLICIES), "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
            for name in ("episode_summaries.jsonl", "collector_step_diagnostics.jsonl", "oracle_gate.json")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis, sort_keys=True))


if __name__ == "__main__":
    main()
