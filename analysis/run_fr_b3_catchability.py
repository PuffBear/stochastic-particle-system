#!/usr/bin/env python3
"""Run the FR-B3 factorial study or its dimensionless rescaling audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Iterable, Mapping

import numpy as np
import yaml

from particle_benchmark.catchability import (
    catchability_groups,
    physical_parameters_from_groups,
    rescale_equivalent_config,
)
from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.io import canonical_json_bytes, sha256_json
from particle_benchmark.runner import (
    EVENT_KEYED_TIE_SCHEME,
    _jsonable,
    _ndarray_sha256,
    _policy_actions,
)


PRIMARY_POLICIES = ("capacity_matched_independent", "shared_summary_v2")
ALLOWED_POLICIES = PRIMARY_POLICIES + ("stationary", "full_state_interception_oracle")


@dataclass(frozen=True)
class StudyCondition:
    condition_id: str
    study: str
    environment: ParticleEnvConfig
    metadata: dict[str, object]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream_checksums(env: ParticleCollectorEnv) -> dict[str, str]:
    assert env.particle_positions is not None
    assert env.collector_positions is not None
    assert env._noise is not None
    assert env._episode_field_kwargs is not None
    assert env.scenario_seed is not None
    initial_state = {
        "particle_positions_sha256": _ndarray_sha256(env.particle_positions),
        "collector_positions_sha256": _ndarray_sha256(env.collector_positions),
    }
    tie_key = {
        "scheme": EVENT_KEYED_TIE_SCHEME,
        "scenario_seed": env.scenario_seed,
        "event_key": ["scenario_seed", "one_based_step_index", "stable_particle_id"],
        "candidate_order": "sorted_stable_collector_ids",
    }
    return {
        "initial_state_sha256": sha256_json(initial_state),
        "brownian_sha256": _ndarray_sha256(env._noise),
        "field_nuisance_sha256": sha256_json(_jsonable(env._episode_field_kwargs)),
        "tie_key_provenance_sha256": sha256_json(tie_key),
    }


def _normalized_final_state_sha256(env: ParticleCollectorEnv) -> str:
    """Checksum a scale-normalized final state with numerical quantization."""

    assert env.particle_positions is not None
    assert env.collector_positions is not None
    assert env.capture_state is not None
    arena = np.asarray(env.config.arena_size, dtype=np.float64)
    payload = {
        "particles": np.round(env.particle_positions / arena, 12).tolist(),
        "collectors": np.round(env.collector_positions / arena, 12).tolist(),
        "capture_owner": env.capture_state.owner.astype(int).tolist(),
    }
    return sha256_json(payload)


def _fixed_environment(protocol: Mapping[str, object]) -> dict[str, object]:
    fixed = dict(protocol["fixed_environment"])
    required = {
        "arena_size",
        "dt",
        "horizon",
        "particle_count",
        "collector_count",
        "sensing_radius",
        "collector_radius",
        "particle_radius",
        "field_family",
        "capture_geometry",
        "nearest_particles_k",
    }
    if set(fixed) != required:
        raise ValueError(f"fixed_environment keys must be exactly {sorted(required)}")
    fixed["arena_size"] = tuple(float(x) for x in fixed["arena_size"])
    return fixed


def _environment_from_groups(
    protocol: Mapping[str, object], *, rho: float, kappa: float, eta: float
) -> ParticleEnvConfig:
    fixed = _fixed_environment(protocol)
    physical = physical_parameters_from_groups(
        rho=rho,
        kappa=kappa,
        eta=eta,
        dt=float(fixed["dt"]),
        arena_size=fixed["arena_size"],
    )
    return ParticleEnvConfig(**fixed, **physical)


def factorial_conditions(protocol: Mapping[str, object]) -> list[StudyCondition]:
    axes = dict(protocol["factorial_axes"])
    if set(axes) != {"rho", "kappa", "eta"}:
        raise ValueError("factorial_axes must contain exactly rho, kappa, and eta")
    levels = {
        key: tuple(float(value) for value in axes[key]) for key in ("rho", "kappa", "eta")
    }
    if any(len(values) < 2 for values in levels.values()):
        raise ValueError("each factorial axis requires at least two levels")
    conditions: list[StudyCondition] = []
    for ri, ki, ei in itertools.product(
        range(len(levels["rho"])), range(len(levels["kappa"])), range(len(levels["eta"]))
    ):
        rho, kappa, eta = levels["rho"][ri], levels["kappa"][ki], levels["eta"][ei]
        environment = _environment_from_groups(
            protocol, rho=rho, kappa=kappa, eta=eta
        )
        measured = catchability_groups(environment)
        requested = np.asarray([rho, kappa, eta])
        recovered = np.asarray([measured.rho, measured.kappa, measured.eta])
        np.testing.assert_allclose(recovered, requested, rtol=1e-12, atol=1e-12)
        conditions.append(
            StudyCondition(
                condition_id=f"r{ri}_k{ki}_e{ei}",
                study="factorial",
                environment=environment,
                metadata={"rho_index": ri, "kappa_index": ki, "eta_index": ei},
            )
        )
    return conditions


def rescaling_conditions(protocol: Mapping[str, object]) -> list[StudyCondition]:
    anchor = {
        key: float(value)
        for key, value in dict(protocol["anchor"]).items()
        if key in {"rho", "kappa", "eta"}
    }
    base = _environment_from_groups(protocol, **anchor)
    audit = dict(protocol["rescaling_audit"])
    conditions: list[StudyCondition] = []
    reference = catchability_groups(base)
    for variant in audit["variants"]:
        variant = dict(variant)
        environment = rescale_equivalent_config(
            base,
            length_scale=float(variant["length_scale"]),
            time_scale=float(variant["time_scale"]),
        )
        measured = catchability_groups(environment)
        np.testing.assert_allclose(
            [measured.rho, measured.kappa, measured.eta],
            [reference.rho, reference.kappa, reference.eta],
            rtol=1e-12,
            atol=1e-12,
        )
        conditions.append(
            StudyCondition(
                condition_id=str(variant["id"]),
                study="rescaling_audit",
                environment=environment,
                metadata={
                    "length_scale": float(variant["length_scale"]),
                    "time_scale": float(variant["time_scale"]),
                },
            )
        )
    return conditions


def load_protocol(path: Path) -> dict[str, object]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "FR-B3-CATCHABILITY":
        raise ValueError("wrong or missing FR-B3 experiment_id")
    if protocol.get("protocol_status") not in {
        "proposed_not_preregistered",
        "registered",
    }:
        raise ValueError("unsupported FR-B3 protocol status")
    policies = tuple(str(x) for x in protocol["policies"])
    if policies != ALLOWED_POLICIES:
        raise ValueError(f"factorial policies must equal {ALLOWED_POLICIES}")
    seeds = tuple(int(x) for x in protocol["seeds"])
    if len(seeds) != len(set(seeds)) or len(seeds) < 8:
        raise ValueError("factorial seeds must be unique with at least eight values")
    return protocol


def rollout(
    condition: StudyCondition, seed: int, policy_id: str
) -> dict[str, object]:
    if policy_id not in ALLOWED_POLICIES:
        raise ValueError(f"unsupported FR-B3 policy: {policy_id}")
    env = ParticleCollectorEnv(condition.environment)
    observations, reset_info = env.reset(seed=seed)
    if reset_info["captured_total"] != 0 or reset_info["tie_scheme"] != EVENT_KEYED_TIE_SCHEME:
        raise RuntimeError("reset or tie provenance violated the FR-B3 contract")
    checksums = _stream_checksums(env)
    path_length = np.zeros(env.config.collector_count, dtype=np.float64)
    capture_count = 0

    for step in range(env.config.horizon):
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
        observations, _, terminated, truncated, info = env.step(actions)
        path_length += np.linalg.norm(env.collector_positions - before, axis=1)
        capture_count += len(info["captures"])
        if info["tie_scheme"] != EVENT_KEYED_TIE_SCHEME:
            raise RuntimeError("tie provenance changed during rollout")
        if terminated or truncated:
            break

    assert env.capture_state is not None
    unique_yield = int(np.count_nonzero(env.capture_state.owner >= 0))
    if unique_yield != capture_count:
        raise RuntimeError("FR-B3 unique-capture accounting failure")
    if env.step_count != env.config.horizon and unique_yield != env.config.particle_count:
        raise RuntimeError("FR-B3 episode ended before the fixed horizon")
    groups = catchability_groups(env.config)
    return {
        "study": condition.study,
        "condition_id": condition.condition_id,
        "condition_metadata": condition.metadata,
        "seed": seed,
        "policy_id": policy_id,
        "unique_team_capture_yield": unique_yield,
        "executed_steps": env.step_count,
        "physical_parameters": {
            "arena_size": list(env.config.arena_size),
            "dt": env.config.dt,
            "diffusion_sigma": env.config.diffusion_sigma,
            "signal_strength": env.config.signal_strength,
            "collector_max_speed": env.config.collector_max_speed,
            "sensing_radius": env.config.sensing_radius,
            "collector_radius": env.config.collector_radius,
        },
        "dimensionless_groups": groups.to_dict(),
        "total_collector_path_length": float(np.sum(path_length)),
        "stream_checksums": checksums,
        "normalized_final_state_sha256": _normalized_final_state_sha256(env),
    }


def _worker(task: tuple[StudyCondition, int, str]) -> dict[str, object]:
    return rollout(*task)


def _validate_paired_streams(rows: Iterable[Mapping[str, object]]) -> None:
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["condition_id"]), int(row["seed"]))
        groups.setdefault(key, []).append(row)
    for key, grouped in groups.items():
        checks = [dict(row["stream_checksums"]) for row in grouped]
        if any(value != checks[0] for value in checks[1:]):
            raise RuntimeError(f"matched streams differ across policies for {key}")


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _study_design(
    protocol: Mapping[str, object], study: str
) -> tuple[list[StudyCondition], tuple[int, ...], tuple[str, ...]]:
    if study == "factorial":
        return (
            factorial_conditions(protocol),
            tuple(int(x) for x in protocol["seeds"]),
            tuple(str(x) for x in protocol["policies"]),
        )
    audit = dict(protocol["rescaling_audit"])
    excluded = {int(x) for x in audit.get("excluded_development_seeds", [])}
    audit_seeds = tuple(int(x) for x in audit["seeds"])
    if excluded.intersection(audit_seeds):
        raise ValueError("development seeds cannot enter the frozen rescaling audit")
    return (
        rescaling_conditions(protocol),
        audit_seeds,
        tuple(str(x) for x in audit["policies"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study", choices=("factorial", "rescaling-audit"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository-commit")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--max-seeds", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    protocol = load_protocol(args.config)
    study_key = args.study.replace("-", "_")
    conditions, seeds, policies = _study_design(protocol, study_key)
    if args.max_cells is not None:
        conditions = conditions[: args.max_cells]
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    design = {
        "experiment_id": protocol["experiment_id"],
        "study": study_key,
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "policies": list(policies),
        "episode_count": len(conditions) * len(seeds) * len(policies),
        "development_limits": {
            "max_cells": args.max_cells,
            "max_seeds": args.max_seeds,
        },
        "conditions": [
            {
                "condition_id": item.condition_id,
                "metadata": item.metadata,
                "environment": _jsonable(asdict(item.environment)),
                "dimensionless_groups": catchability_groups(item.environment).to_dict(),
            }
            for item in conditions
        ],
    }
    if args.dry_run:
        print(json.dumps(design, indent=2, sort_keys=True))
        return
    complete_design = args.max_cells is None and args.max_seeds is None
    if complete_design and protocol["protocol_status"] != "registered":
        raise RuntimeError(
            "full frozen-seed execution requires protocol_status=registered; "
            "use development limits before external registration"
        )
    if args.output is None or args.repository_commit is None:
        parser.error("--output and --repository-commit are required unless --dry-run is used")
    if len(args.repository_commit) != 40:
        raise ValueError("--repository-commit must be a full 40-character SHA")
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")

    tasks = list(itertools.product(conditions, seeds, policies))
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    if args.jobs == 1:
        for index, task in enumerate(tasks, start=1):
            rows.append(_worker(task))
            print(f"completed {index}/{len(tasks)}", file=sys.stderr, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_worker, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(f"completed {index}/{len(tasks)}", file=sys.stderr, flush=True)
    rows.sort(key=lambda row: (str(row["condition_id"]), int(row["seed"]), str(row["policy_id"])))
    _validate_paired_streams(rows)
    runtime_seconds = time.perf_counter() - started

    args.output.mkdir(parents=True, exist_ok=False)
    summaries_path = args.output / "episode_summaries.jsonl"
    design_path = args.output / "design.json"
    _write_new(summaries_path, b"".join(canonical_json_bytes(row) for row in rows))
    _write_new(design_path, canonical_json_bytes(design))
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        root / "analysis" / "run_fr_b3_catchability.py",
        root / "src" / "particle_benchmark" / "catchability.py",
        args.config.resolve(),
    )
    manifest = {
        "experiment_id": protocol["experiment_id"],
        "study": study_key,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": protocol["protocol_status"],
        "complete_frozen_design": complete_design,
        "repository": {
            "full_name": "PuffBear/stochastic-particle-system",
            "branch": "fr-b3-catchability-benchmark",
            "commit_sha": args.repository_commit,
        },
        "runtime": {
            "command": " ".join(sys.argv),
            "seconds": runtime_seconds,
            "jobs": args.jobs,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "source_snapshot": {
            path.relative_to(root).as_posix(): _file_sha256(path) for path in source_paths
        },
        "artifacts": {
            summaries_path.name: _file_sha256(summaries_path),
            design_path.name: _file_sha256(design_path),
        },
    }
    _write_new(args.output / "manifest.json", canonical_json_bytes(manifest))
    print(json.dumps({"output": str(args.output), "episodes": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
