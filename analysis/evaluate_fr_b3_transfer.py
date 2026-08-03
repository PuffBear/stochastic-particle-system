#!/usr/bin/env python3
"""Deterministically evaluate registered FR-B3 learned-policy bundles."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from typing import Mapping

import numpy as np
import yaml

from particle_benchmark.catchability import (
    catchability_groups,
    rescale_equivalent_config,
)
from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.io import canonical_json_bytes


EXPECTED_EXPERIMENT = "FR-B3-LEARNED-TRANSFER"
EXPECTED_BRANCH = "fr-b3-catchability-benchmark"


def load_protocol(path: Path) -> dict[str, object]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != EXPECTED_EXPERIMENT:
        raise ValueError("wrong learned-transfer experiment config")
    return protocol


def canonical_config(protocol: Mapping[str, object]) -> ParticleEnvConfig:
    environment = dict(protocol["canonical_environment"])
    environment["arena_size"] = tuple(float(x) for x in environment["arena_size"])
    return ParticleEnvConfig(**environment)


def design_summary(protocol: Mapping[str, object]) -> dict[str, object]:
    architectures = tuple(str(item) for item in protocol["architectures"])
    representations = tuple(str(item) for item in protocol["representations"])
    training_seeds = tuple(int(item) for item in protocol["training_seeds"])
    evaluation_seeds = tuple(int(item) for item in protocol["evaluation_seeds"])
    tuning_seeds = tuple(int(item) for item in protocol["tuning_seeds"])
    scales = tuple(dict(item) for item in protocol["evaluation_scales"])
    if architectures != ("ippo", "commnet"):
        raise ValueError("architectures must be exactly IPPO and CommNet")
    if representations != ("raw_physical", "dimensionless"):
        raise ValueError("representations must be raw physical and dimensionless")
    if training_seeds != tuple(range(8301, 8306)):
        raise ValueError("training seed firewall must be exactly 8301-8305")
    if evaluation_seeds != tuple(range(8601, 8665)):
        raise ValueError("evaluation seed firewall must be exactly 8601-8664")
    if tuning_seeds != tuple(range(8701, 8717)):
        raise ValueError("tuning seed firewall must be exactly 8701-8716")
    if any(len(panel) != len(set(panel)) for panel in (training_seeds, evaluation_seeds, tuning_seeds)):
        raise ValueError("a learned-transfer seed panel contains duplicates")
    if set(training_seeds) & set(evaluation_seeds):
        raise ValueError("training and evaluation seed panels overlap")
    if (set(training_seeds) | set(evaluation_seeds)) & set(tuning_seeds):
        raise ValueError("tuning seeds overlap a final seed panel")
    expected_scales = (
        ("canonical", 1.0, 1.0),
        ("length_x2", 2.0, 1.0),
        ("time_x4", 1.0, 4.0),
        ("mixed_half", 0.5, 0.25),
    )
    actual_scales = tuple(
        (
            str(item["id"]),
            float(item["length_scale"]),
            float(item["time_scale"]),
        )
        for item in scales
    )
    if actual_scales != expected_scales:
        raise ValueError("evaluation scales differ from the candidate protocol")
    expected_environment = ParticleEnvConfig(
        arena_size=(1.0, 1.0),
        dt=0.02,
        horizon=67,
        particle_count=256,
        collector_count=4,
        diffusion_sigma=0.06,
        collector_max_speed=0.12,
        sensing_radius=0.16,
        collector_radius=0.012,
        particle_radius=0.0,
        field_family="uniform",
        signal_strength=0.06,
        capture_geometry="fixed",
        nearest_particles_k=32,
        include_particle_velocity=True,
        include_teammates=True,
    )
    if canonical_config(protocol) != expected_environment:
        raise ValueError("canonical environment differs from the executed SPS-C03 anchor")
    evaluation = dict(protocol["evaluation"])
    if evaluation != {
        "action_rule": "deterministic_mean",
        "update_normalization_at_evaluation": False,
        "target_scale_fine_tuning": False,
        "commnet_zero_message_ablation": True,
    }:
        raise ValueError("evaluation intervention differs from the candidate contract")
    if protocol.get("training_scale") != "canonical":
        raise ValueError("training scale must be canonical")
    bundle_count = len(architectures) * len(representations) * len(training_seeds)
    primary_episodes = bundle_count * len(scales) * len(evaluation_seeds)
    commnet_bundle_count = len(representations) * len(training_seeds)
    ablation_episodes = commnet_bundle_count * len(scales) * len(evaluation_seeds)
    return {
        "experiment_id": EXPECTED_EXPERIMENT,
        "protocol_status": protocol["protocol_status"],
        "training_run_count": bundle_count,
        "required_bundle_count": bundle_count,
        "primary_evaluation_episode_count": primary_episodes,
        "commnet_ablation_episode_count": ablation_episodes,
        "total_evaluation_episode_count": primary_episodes + ablation_episodes,
        "training_seeds": list(training_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "tuning_seeds": list(tuning_seeds),
        "architectures": list(architectures),
        "representations": list(representations),
        "scales": [dict(item) for item in scales],
        "action_rule": "deterministic_mean",
        "normalization_updates_at_evaluation": False,
    }


def _normalized_final_state(env: ParticleCollectorEnv) -> str:
    assert env.particle_positions is not None
    assert env.collector_positions is not None
    assert env.capture_state is not None
    arena = np.asarray(env.config.arena_size, dtype=np.float64)
    payload = {
        "particles": np.round(env.particle_positions / arena, 12).tolist(),
        "collectors": np.round(env.collector_positions / arena, 12).tolist(),
        "capture_owner": env.capture_state.owner.astype(int).tolist(),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def evaluate_bundle_episode(
    bundle: object,
    *,
    environment: ParticleEnvConfig,
    seed: int,
    communication_ablated: bool,
) -> dict[str, object]:
    """Evaluate one development or registered episode without policy sampling."""

    from particle_benchmark.marl.representations import (
        AdaptedObservationEnv,
        FRB3ObservationAdapter,
    )
    from particle_benchmark.marl.transfer import deterministic_actions

    adapter = FRB3ObservationAdapter(environment, bundle.representation)
    wrapped = AdaptedObservationEnv(
        ParticleCollectorEnv(environment), adapter, bundle.standardizer
    )
    observations, _ = wrapped.reset(seed=seed)
    done = False
    while not done:
        actions = deterministic_actions(
            bundle,
            observations,
            communication_ablated=communication_ablated,
        )
        observations, _, terminated, truncated, info = wrapped.step(actions)
        done = terminated or truncated
    return {
        "seed": seed,
        "evaluation_mode": (
            "zero_message" if communication_ablated else "full_policy"
        ),
        "unique_team_capture_yield": int(info["captured_total"]),
        "executed_steps": int(info["step"]),
        "normalized_final_state_sha256": _normalized_final_state(wrapped.env),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _verify_repository(root: Path) -> tuple[str, str]:
    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"wrong branch: {branch!r}")
    if len(commit) != 40:
        raise RuntimeError("HEAD is not a full commit SHA")
    if status:
        raise RuntimeError("tracked worktree changes are present")
    return branch, commit


def _bundle_key(bundle: object) -> tuple[str, str, int]:
    return bundle.architecture, bundle.representation, bundle.training_seed


def run_registered_evaluation(
    protocol: Mapping[str, object],
    *,
    bundle_dirs: list[Path],
    repository_root: Path,
) -> dict[str, object]:
    """Require all 20 bundles, then evaluate the complete frozen Cartesian set."""

    if protocol.get("protocol_status") != "registered":
        raise RuntimeError("learned-transfer protocol is not registered")
    summary = design_summary(protocol)
    if len(bundle_dirs) != int(summary["required_bundle_count"]):
        raise ValueError("registered evaluation requires exactly 20 policy bundles")
    branch, commit = _verify_repository(repository_root)
    base = canonical_config(protocol)

    from particle_benchmark.marl.transfer import load_policy_bundle

    bundles = [
        load_policy_bundle(path, canonical_config=base) for path in bundle_dirs
    ]
    expected_keys = set(
        itertools.product(
            summary["architectures"],
            summary["representations"],
            summary["training_seeds"],
        )
    )
    observed_keys = [_bundle_key(bundle) for bundle in bundles]
    if len(set(observed_keys)) != len(observed_keys):
        raise ValueError("duplicate architecture-representation-training-seed bundle")
    if set(observed_keys) != expected_keys:
        raise ValueError("policy bundles do not cover the frozen training matrix")

    rows: list[dict[str, object]] = []
    scales = [dict(item) for item in protocol["evaluation_scales"]]
    for bundle in sorted(bundles, key=_bundle_key):
        for scale in scales:
            environment = rescale_equivalent_config(
                base,
                length_scale=float(scale["length_scale"]),
                time_scale=float(scale["time_scale"]),
            )
            modes = (False, True) if bundle.architecture == "commnet" else (False,)
            for communication_ablated in modes:
                for seed in summary["evaluation_seeds"]:
                    result = evaluate_bundle_episode(
                        bundle,
                        environment=environment,
                        seed=int(seed),
                        communication_ablated=communication_ablated,
                    )
                    rows.append(
                        {
                            "architecture": bundle.architecture,
                            "representation": bundle.representation,
                            "training_seed": bundle.training_seed,
                            "checkpoint_episode": bundle.checkpoint_episode,
                            "scale_id": scale["id"],
                            "dimensionless_groups": catchability_groups(
                                environment
                            ).to_dict(),
                            **result,
                        }
                    )
    if len(rows) != int(summary["total_evaluation_episode_count"]):
        raise RuntimeError("evaluation row count differs from frozen design")
    return {
        "experiment_id": EXPECTED_EXPERIMENT,
        "protocol_status": "registered",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": summary,
        "repository": {"branch": branch, "commit_sha": commit},
        "canonical_environment": asdict(base),
        "bundles": [dict(bundle.metadata) for bundle in bundles],
        "episode_results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, action="append", default=[])
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.config)
    summary = design_summary(protocol)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if protocol.get("protocol_status") != "registered":
        raise SystemExit(
            "FR-B3 learned-transfer protocol is not registered; no candidate seed ran"
        )
    if args.output is None:
        parser.error("--output is required outside --dry-run")
    if args.output.exists():
        raise FileExistsError(f"immutable evaluation output exists: {args.output}")
    report = run_registered_evaluation(
        protocol,
        bundle_dirs=args.bundle_dir,
        repository_root=args.repository_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(report["episode_results"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
