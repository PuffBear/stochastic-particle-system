"""Deterministic matched null/signal episode runner.

The runner treats counterfactual pairing as a correctness contract.  A pair is
rejected unless its two environment configurations differ only in
``signal_strength`` and reset exposes identical initialization, Brownian
forcing, field nuisance, policy-randomness provenance, and event-keyed tie
provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
import yaml

from .dynamics.fields import field_velocity
from .environment import ParticleCollectorEnv, ParticleEnvConfig
from .io import (
    PreparedArtifact,
    canonical_json_bytes,
    load_schema,
    prepare_json,
    prepare_jsonl,
    sha256_json,
    write_immutable_artifacts,
)
from .observations import LocalObservation
from .policies import (
    capacity_matched_velocity_controller,
    capacity_matched_velocity_controller_v2,
    coverage_policy,
    density_greedy_policy,
    full_state_interception_oracle,
    local_flow_v1_policy,
    privileged_field_policy,
    random_action_tensor,
    stationary_policy,
)


SUPPORTED_POLICIES = frozenset(
    {
        "stationary",
        "pregenerated_random",
        "coverage",
        "density_greedy",
        "local_flow_v1",
        "privileged_upstream_oracle",
        "true_field_upstream_control",
        "full_state_interception_oracle",
        "capacity_matched_independent",
        "shared_summary",
        "shared_summary_v2",
    }
)
EVENT_KEYED_TIE_SCHEME = "event_keyed_seed_step_particle_v1"


@dataclass(frozen=True)
class RepositoryProvenance:
    """Version-control provenance required in every immutable run manifest."""

    full_name: str
    branch: str
    commit_sha: str
    dirty: bool

    def __post_init__(self) -> None:
        if not self.full_name or not self.branch:
            raise ValueError("repository name and branch must be non-empty")
        if re.fullmatch(r"[0-9a-f]{40}", self.commit_sha) is None:
            raise ValueError("commit_sha must be a full lowercase 40-character SHA")


@dataclass(frozen=True)
class PairExperimentConfig:
    """Frozen inputs for one matched null/signal pair."""

    experiment_id: str
    scenario_seed: int
    signal_strength: float
    policy_id: str
    environment: ParticleEnvConfig
    repository: RepositoryProvenance
    created_at_utc: str
    config_path: str
    run_command: str
    policy_config: dict[str, object] = field(default_factory=dict)
    stop_after_first_interception: bool = False

    def __post_init__(self) -> None:
        if not self.experiment_id or self.scenario_seed < 0:
            raise ValueError("experiment_id must be non-empty and seed non-negative")
        if not np.isfinite(self.signal_strength) or self.signal_strength < 0:
            raise ValueError("signal_strength must be finite and non-negative")
        if self.policy_id not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported policy_id: {self.policy_id}")
        if self.environment.capture_geometry != "fixed":
            raise ValueError("primary matched runner supports fixed geometry only")
        if not self.config_path or not self.run_command:
            raise ValueError("config_path and run_command must be non-empty")
        try:
            parsed = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at_utc must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("created_at_utc must contain an explicit timezone")
        allowed = {"sweep_period"} if self.policy_id == "coverage" else set()
        if self.policy_id == "full_state_interception_oracle":
            allowed = {"receding_horizon"}
        unknown = set(self.policy_config) - allowed
        if unknown:
            raise ValueError(f"unexpected policy config keys: {sorted(unknown)}")


@dataclass(frozen=True)
class PairArtifactPaths:
    null_trajectory: Path
    signal_trajectory: Path
    summary: Path
    manifest: Path


@dataclass(frozen=True)
class PairRunResult:
    """In-memory summary and paths produced by a completed paired run."""

    run_id: str
    summary: dict[str, Any]
    manifest: dict[str, Any]
    paths: PairArtifactPaths


def _ndarray_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes(list(contiguous.shape), newline=False))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value: object) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def validate_paired_environment_configs(
    null_config: ParticleEnvConfig, signal_config: ParticleEnvConfig
) -> None:
    """Reject hidden treatment differences before simulation starts."""
    null_values = asdict(null_config)
    signal_values = asdict(signal_config)
    differences = {
        key
        for key in null_values
        if _jsonable(null_values[key]) != _jsonable(signal_values[key])
    }
    if differences - {"signal_strength"}:
        raise ValueError(
            "matched environment configs differ outside signal_strength: "
            + ", ".join(sorted(differences - {"signal_strength"}))
        )
    if null_config.signal_strength != 0.0:
        raise ValueError("null condition signal_strength must equal zero")
    if signal_config.signal_strength < 0.0:
        raise ValueError("signal condition signal_strength must be non-negative")


def _pairing_inputs(
    null_env: ParticleCollectorEnv,
    signal_env: ParticleCollectorEnv,
    *,
    scenario_seed: int,
    policy_randomness_sha256: str,
) -> dict[str, str]:
    assert null_env.particle_positions is not None
    assert signal_env.particle_positions is not None
    assert null_env.collector_positions is not None
    assert signal_env.collector_positions is not None
    assert null_env._noise is not None
    assert signal_env._noise is not None
    assert null_env._episode_field_kwargs is not None
    assert signal_env._episode_field_kwargs is not None

    if not np.array_equal(null_env.particle_positions, signal_env.particle_positions):
        raise RuntimeError("pairing mismatch: initial particle positions")
    if not np.array_equal(null_env.collector_positions, signal_env.collector_positions):
        raise RuntimeError("pairing mismatch: initial collector positions")
    if not np.array_equal(null_env._noise, signal_env._noise):
        raise RuntimeError("pairing mismatch: Brownian tensor")
    if _jsonable(null_env._episode_field_kwargs) != _jsonable(signal_env._episode_field_kwargs):
        raise RuntimeError("pairing mismatch: field nuisance")
    if null_env.scenario_seed != scenario_seed or signal_env.scenario_seed != scenario_seed:
        raise RuntimeError("pairing mismatch: scenario seed provenance")
    if null_env.tie_scheme != EVENT_KEYED_TIE_SCHEME or signal_env.tie_scheme != EVENT_KEYED_TIE_SCHEME:
        raise RuntimeError("pairing mismatch: fixed geometry must use event-keyed ties")

    initial_state = {
        "particle_positions_sha256": _ndarray_sha256(null_env.particle_positions),
        "collector_positions_sha256": _ndarray_sha256(null_env.collector_positions),
    }
    tie_key = {
        "scheme": EVENT_KEYED_TIE_SCHEME,
        "scenario_seed": scenario_seed,
        "event_key": ["scenario_seed", "one_based_step_index", "stable_particle_id"],
        "candidate_order": "sorted_stable_collector_ids",
    }
    return {
        "initial_state_sha256": sha256_json(initial_state),
        "brownian_sha256": _ndarray_sha256(null_env._noise),
        "field_nuisance_sha256": sha256_json(_jsonable(null_env._episode_field_kwargs)),
        "policy_randomness_sha256": policy_randomness_sha256,
        "tie_key_provenance_sha256": sha256_json(tie_key),
    }


def _policy_randomness(
    policy_id: str, *, scenario_seed: int, horizon: int, collector_count: int
) -> tuple[np.ndarray | None, str]:
    if policy_id == "pregenerated_random":
        actions = random_action_tensor(
            scenario_seed, horizon=horizon, collector_count=collector_count
        )
        return actions, _ndarray_sha256(actions)
    return None, sha256_json({"policy_id": policy_id, "randomness": "none"})


def _policy_actions(
    policy_id: str,
    observations: tuple[LocalObservation, ...],
    env: ParticleCollectorEnv,
    *,
    step: int,
    random_actions: np.ndarray | None,
    policy_config: Mapping[str, object],
) -> np.ndarray:
    if policy_id == "stationary":
        return stationary_policy(env.config.collector_count)
    if policy_id == "pregenerated_random":
        assert random_actions is not None
        return random_actions[step].copy()
    if policy_id == "coverage":
        return coverage_policy(
            observations,
            step=step,
            sweep_period=int(policy_config.get("sweep_period", 100)),
        )
    if policy_id == "density_greedy":
        return density_greedy_policy(observations)
    if policy_id == "local_flow_v1":
        return local_flow_v1_policy(observations)
    if policy_id in {"privileged_upstream_oracle", "true_field_upstream_control"}:
        assert env.collector_positions is not None
        assert env._episode_field_kwargs is not None
        return privileged_field_policy(
            env.collector_positions,
            field_family=env.config.field_family,
            signal_strength=env.config.signal_strength,
            field_kwargs=env._episode_field_kwargs,
        )
    if policy_id == "full_state_interception_oracle":
        assert env.collector_positions is not None
        assert env.particle_positions is not None
        assert env._episode_field_kwargs is not None
        assert env.capture_state is not None
        true_motion = field_velocity(
            env.particle_positions,
            env.config.field_family,
            env.config.signal_strength,
            **env._episode_field_kwargs,
        )
        return full_state_interception_oracle(
            env.collector_positions,
            env.particle_positions,
            true_motion,
            env.capture_state.owner < 0,
            collector_max_speed=env.config.collector_max_speed,
            receding_horizon=float(policy_config.get("receding_horizon", 2.0)),
        )
    if policy_id == "capacity_matched_independent":
        return capacity_matched_velocity_controller(observations, shared=False)
    if policy_id == "shared_summary":
        return capacity_matched_velocity_controller(observations, shared=True)
    if policy_id == "shared_summary_v2":
        return capacity_matched_velocity_controller_v2(observations, shared=True)
    raise AssertionError(f"unreachable policy {policy_id}")


def _trajectory_record(
    *,
    run_id: str,
    condition: str,
    policy_id: str,
    scenario_seed: int,
    env: ParticleCollectorEnv,
    actions: np.ndarray,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    assert env.particle_positions is not None
    assert env.collector_positions is not None
    assert env.capture_state is not None
    return {
        "schema_version": 1,
        "run_id": run_id,
        "scenario_seed": scenario_seed,
        "condition": condition,
        "policy_id": policy_id,
        "step": int(info["step"]),
        "particle_positions": env.particle_positions.tolist(),
        "collector_positions": env.collector_positions.tolist(),
        "collector_actions": actions.tolist(),
        "capture_owner": env.capture_state.owner.tolist(),
        "capture_events": [
            {"particle_id": int(particle_id), "collector_id": int(collector_id)}
            for particle_id, collector_id in info["captures"]
        ],
        "tie_break_scheme": str(info["tie_scheme"]),
        "tie_provenance": _jsonable(info["tie_provenance"]),
        "contact_time_fractions": _jsonable(info["contact_time_fractions"]),
        "contact_model": str(info["contact_model"]),
        "guarded_reflection_pairs": int(info["guarded_reflection_pairs"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _zero_signal_records_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    ignored = {"condition"}
    return {k: v for k, v in left.items() if k not in ignored} == {
        k: v for k, v in right.items() if k not in ignored
    }


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not slug:
        raise ValueError("value cannot be converted to a safe filename")
    return slug


def _schema_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas"


def run_matched_pair(config: PairExperimentConfig, output_dir: str | Path) -> PairRunResult:
    """Run, verify, validate, and immutably write one paired episode."""
    null_config = replace(config.environment, signal_strength=0.0)
    signal_config = replace(config.environment, signal_strength=config.signal_strength)
    validate_paired_environment_configs(null_config, signal_config)

    frozen_config = {
        "experiment_id": config.experiment_id,
        "scenario_seed": config.scenario_seed,
        "signal_strength": config.signal_strength,
        "policy_id": config.policy_id,
        "policy_config": _jsonable(config.policy_config),
        "environment": _jsonable(asdict(config.environment)),
        "repository_commit_sha": config.repository.commit_sha,
        "stop_after_first_interception": config.stop_after_first_interception,
    }
    config_sha256 = sha256_json(frozen_config)
    run_id = _safe_slug(
        f"{config.experiment_id}-{config.policy_id}-seed{config.scenario_seed}-"
        f"a{config.signal_strength:.12g}-{config_sha256[:12]}-{config.repository.commit_sha[:12]}"
    )

    random_actions, policy_randomness_sha256 = _policy_randomness(
        config.policy_id,
        scenario_seed=config.scenario_seed,
        horizon=config.environment.horizon,
        collector_count=config.environment.collector_count,
    )
    null_env = ParticleCollectorEnv(null_config)
    signal_env = ParticleCollectorEnv(signal_config)
    null_observations, null_reset = null_env.reset(seed=config.scenario_seed)
    signal_observations, signal_reset = signal_env.reset(seed=config.scenario_seed)
    if null_reset["scenario_seed"] != signal_reset["scenario_seed"]:
        raise RuntimeError("pairing mismatch: reset scenario seeds")
    if null_reset["tie_scheme"] != signal_reset["tie_scheme"]:
        raise RuntimeError("pairing mismatch: reset tie schemes")
    input_checksums = _pairing_inputs(
        null_env,
        signal_env,
        scenario_seed=config.scenario_seed,
        policy_randomness_sha256=policy_randomness_sha256,
    )

    records: dict[str, list[dict[str, Any]]] = {"null": [], "signal": []}
    observations = {"null": null_observations, "signal": signal_observations}
    environments = {"null": null_env, "signal": signal_env}
    done = {"null": False, "signal": False}
    zero_signal_identity_checked = config.signal_strength == 0.0
    for step in range(config.environment.horizon):
        step_records: dict[str, dict[str, Any]] = {}
        for condition in ("null", "signal"):
            if done[condition]:
                continue
            env = environments[condition]
            actions = _policy_actions(
                config.policy_id,
                observations[condition],
                env,
                step=step,
                random_actions=random_actions,
                policy_config=config.policy_config,
            )
            next_observations, _, terminated, truncated, info = env.step(actions)
            if info["scenario_seed"] != config.scenario_seed:
                raise RuntimeError("pairing mismatch: step scenario seed provenance")
            if info["tie_scheme"] != EVENT_KEYED_TIE_SCHEME:
                raise RuntimeError("pairing mismatch: non-event-keyed tie provenance")
            record = _trajectory_record(
                run_id=run_id,
                condition=condition,
                policy_id=config.policy_id,
                scenario_seed=config.scenario_seed,
                env=env,
                actions=actions,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            records[condition].append(record)
            step_records[condition] = record
            observations[condition] = next_observations
            done[condition] = terminated or truncated or bool(
                config.stop_after_first_interception
                and env.first_contact_step is not None
            )

        if zero_signal_identity_checked:
            if set(step_records) != {"null", "signal"}:
                raise RuntimeError("zero-signal identity failed: episode lengths differ")
            if not _zero_signal_records_equal(step_records["null"], step_records["signal"]):
                raise RuntimeError(f"zero-signal identity failed at step {step + 1}")
        if all(done.values()):
            break

    horizon = config.environment.horizon
    null_first = null_env.first_contact_step or horizon + 1
    signal_first = signal_env.first_contact_step or horizon + 1
    null_last = records["null"][-1]
    signal_last = records["signal"][-1]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "scenario_seed": config.scenario_seed,
        "policy_id": config.policy_id,
        "signal_strength": config.signal_strength,
        "horizon": horizon,
        "null": {
            "first_interception_step": null_first,
            "censored": null_env.first_contact_step is None,
            "captured_total": int(sum(owner >= 0 for owner in null_last["capture_owner"])),
            "trajectory_steps": len(records["null"]),
            "tie_decision_count": sum(len(row["tie_provenance"]) for row in records["null"]),
            "guarded_reflection_pairs": sum(row["guarded_reflection_pairs"] for row in records["null"]),
        },
        "signal": {
            "first_interception_step": signal_first,
            "censored": signal_env.first_contact_step is None,
            "captured_total": int(sum(owner >= 0 for owner in signal_last["capture_owner"])),
            "trajectory_steps": len(records["signal"]),
            "tie_decision_count": sum(len(row["tie_provenance"]) for row in records["signal"]),
            "guarded_reflection_pairs": sum(row["guarded_reflection_pairs"] for row in records["signal"]),
        },
        "matched_first_interception_gain": (null_first - signal_first) / horizon,
        "stopped_after_first_interception": config.stop_after_first_interception,
        "zero_signal_identity_checked": zero_signal_identity_checked,
        "input_checksums": input_checksums,
    }

    schemas = _schema_directory()
    trajectory_schema = load_schema(schemas / "trajectory_step.schema.json")
    summary_schema = load_schema(schemas / "pair_summary.schema.json")
    manifest_schema = load_schema(schemas / "run_manifest.schema.json")
    null_artifact = prepare_jsonl(
        f"{run_id}.null.trajectory.jsonl", records["null"], trajectory_schema
    )
    signal_artifact = prepare_jsonl(
        f"{run_id}.signal.trajectory.jsonl", records["signal"], trajectory_schema
    )
    summary_artifact = prepare_json(f"{run_id}.summary.json", summary, summary_schema)
    data_artifacts = (null_artifact, signal_artifact, summary_artifact)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "created_at_utc": config.created_at_utc,
        "repository": asdict(config.repository),
        "environment": {
            "config_path": config.config_path,
            "config_sha256": config_sha256,
        },
        "pairing": {
            "scenario_seed": config.scenario_seed,
            "conditions": ["null", "signal"],
            "pre_generated_noise": True,
            "shared_initial_state": True,
            "difference_only_signal_strength": True,
            "zero_signal_identity_checked": zero_signal_identity_checked,
            **input_checksums,
        },
        "policy": {
            "policy_id": config.policy_id,
            "config": _jsonable(config.policy_config),
            "privileged": config.policy_id in {
                "privileged_upstream_oracle",
                "true_field_upstream_control",
                "full_state_interception_oracle",
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "command": config.run_command,
        },
        "artifacts": [
            {
                "path": artifact.filename,
                "sha256": artifact.sha256,
                "media_type": artifact.media_type,
                "bytes": len(artifact.payload),
            }
            for artifact in data_artifacts
        ],
    }
    manifest_artifact: PreparedArtifact = prepare_json(
        f"{run_id}.manifest.json", manifest, manifest_schema
    )
    written = write_immutable_artifacts(
        output_dir, (*data_artifacts, manifest_artifact)
    )
    paths = PairArtifactPaths(
        null_trajectory=written[0],
        signal_trajectory=written[1],
        summary=written[2],
        manifest=written[3],
    )
    return PairRunResult(run_id=run_id, summary=summary, manifest=manifest, paths=paths)


def _environment_from_mapping(value: Mapping[str, Any]) -> ParticleEnvConfig:
    expected = set(ParticleEnvConfig.__dataclass_fields__)
    extras = set(value) - expected
    if extras:
        raise ValueError(f"unknown environment keys: {sorted(extras)}")
    converted = dict(value)
    if "arena_size" in converted:
        converted["arena_size"] = tuple(converted["arena_size"])
    return ParticleEnvConfig(**converted)


def load_pair_config(path: str | Path) -> PairExperimentConfig:
    """Load a strict YAML/JSON pair config without implicit scientific defaults."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pair config root must be a mapping")
    required = {
        "experiment_id",
        "scenario_seed",
        "signal_strength",
        "policy_id",
        "environment",
        "repository",
        "created_at_utc",
        "run_command",
    }
    optional = {"policy_config", "stop_after_first_interception"}
    missing = required - set(raw)
    extras = set(raw) - required - optional
    if missing or extras:
        raise ValueError(f"config keys invalid; missing={sorted(missing)}, extra={sorted(extras)}")
    repository = raw["repository"]
    environment = raw["environment"]
    if not isinstance(repository, dict) or not isinstance(environment, dict):
        raise ValueError("repository and environment must be mappings")
    return PairExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        scenario_seed=int(raw["scenario_seed"]),
        signal_strength=float(raw["signal_strength"]),
        policy_id=str(raw["policy_id"]),
        environment=_environment_from_mapping(environment),
        repository=RepositoryProvenance(**repository),
        created_at_utc=str(raw["created_at_utc"]),
        config_path=config_path.as_posix(),
        run_command=str(raw["run_command"]),
        policy_config=dict(raw.get("policy_config", {})),
        stop_after_first_interception=bool(raw.get("stop_after_first_interception", False)),
    )


def run_config_file(path: str | Path, output_dir: str | Path) -> PairRunResult:
    """Deterministically execute a strict pair configuration file."""
    return run_matched_pair(load_pair_config(path), output_dir)
