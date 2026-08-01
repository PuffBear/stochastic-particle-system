"""Deterministic self-only versus all-to-all aggregation-pair runner.

This module is deliberately separate from :mod:`particle_benchmark.runner`,
whose null-versus-signal semantics and legacy artifacts remain unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import itertools
import json
import math
import platform
from pathlib import Path
import re
from typing import Any, Literal, Mapping

import numpy as np
import yaml

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
from .metrics.coordination import (
    global_minus_local_estimator_risk,
    sensed_summary_covariances,
)
from .policies import CapacityMatchedDecision, capacity_matched_velocity_decision
from .runner import RepositoryProvenance


AggregationMode = Literal["independent", "all_to_all"]
Arm = Literal["self_only", "all_to_all"]
ARMS: tuple[Arm, Arm] = ("self_only", "all_to_all")
EVENT_KEYED_TIE_SCHEME = "event_keyed_seed_step_particle_v1"
FROZEN_PHYSICAL_ENDPOINT_SECONDS = 1.34
FROZEN_TIMESTEP_HORIZONS: dict[float, int] = {
    0.02: 67,
    0.01: 134,
    0.005: 268,
}
FROZEN_DIAGNOSTIC_THRESHOLDS = {
    "minimum_arm_eta_analytic_eligibility_rate": 0.80,
    "maximum_absolute_fraction_difference": 0.10,
    "minimum_individual_seed_arm_analytic_eligibility_rate": 0.50,
    "maximum_clipped_velocity_component_rate": 0.01,
    "maximum_own_empty_agent_row_rate": 0.05,
    "maximum_fallback_agent_row_rate": 0.05,
    "maximum_reflected_valid_slot_rate": 0.05,
    "maximum_collector_reflected_action_transition_rate": 0.05,
}


@dataclass(frozen=True)
class AggregationPairExperimentConfig:
    experiment_id: str
    scenario_seed: int
    environment: ParticleEnvConfig
    repository: RepositoryProvenance
    created_at_utc: str
    config_path: str
    run_command: str
    arm_modes: dict[str, AggregationMode] = field(
        default_factory=lambda: {
            "self_only": "independent",
            "all_to_all": "all_to_all",
        }
    )

    def __post_init__(self) -> None:
        if not self.experiment_id or self.scenario_seed < 0:
            raise ValueError("experiment_id must be non-empty and seed non-negative")
        try:
            parsed = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at_utc must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("created_at_utc must contain an explicit timezone")
        if not self.config_path or not self.run_command:
            raise ValueError("config_path and run_command must be non-empty")
        if set(self.arm_modes) != set(ARMS) or any(
            mode not in {"independent", "all_to_all"}
            for mode in self.arm_modes.values()
        ):
            raise ValueError("arm_modes must define self_only and all_to_all")
        canonical = {
            "self_only": "independent",
            "all_to_all": "all_to_all",
        }
        if self.arm_modes != canonical and len(set(self.arm_modes.values())) != 1:
            raise ValueError("only the canonical contrast or a same-mode identity control is allowed")
        env = self.environment
        if env.collector_count != 4:
            raise ValueError("SPS-C04 aggregation runner requires exactly four collectors")
        if env.horizon > 8 and not any(
            math.isclose(env.dt, dt, rel_tol=0.0, abs_tol=1e-12)
            and env.horizon == horizon
            for dt, horizon in FROZEN_TIMESTEP_HORIZONS.items()
        ):
            raise ValueError(
                "horizons above 8 are allowed only for the frozen T=1.34 "
                "timestep mappings"
            )
        if env.field_family != "periodic_gaussian" or env.signal_strength <= 0:
            raise ValueError("aggregation runner requires the same nonzero periodic Gaussian field")
        if env.capture_geometry != "fixed":
            raise ValueError("aggregation runner requires fixed capture geometry")
        if not env.include_particle_velocity or env.include_teammates:
            raise ValueError("particle velocities must be enabled and teammates disabled")


@dataclass(frozen=True)
class AggregationPairArtifactPaths:
    self_only_diagnostics: Path
    all_to_all_diagnostics: Path
    summary: Path
    manifest: Path


@dataclass(frozen=True)
class AggregationPairRunResult:
    run_id: str
    summary: dict[str, Any]
    manifest: dict[str, Any]
    paths: AggregationPairArtifactPaths


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes(list(contiguous.shape), newline=False))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    if not slug:
        raise ValueError("value cannot be converted to a safe filename")
    return slug


def _schema_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas"


def _state_sha256(env: ParticleCollectorEnv) -> str:
    assert env.particle_positions is not None
    assert env.collector_positions is not None
    assert env.capture_state is not None
    assert env._particle_velocities is not None
    assert env._velocity_valid is not None
    return sha256_json(
        {
            "particles": _array_sha256(env.particle_positions),
            "collectors": _array_sha256(env.collector_positions),
            "owners": _array_sha256(env.capture_state.owner),
            "apparent_velocity": _array_sha256(env._particle_velocities),
            "velocity_valid": _array_sha256(env._velocity_valid),
        }
    )


def _observations_equal(left: tuple[Mapping[str, np.ndarray], ...], right: tuple[Mapping[str, np.ndarray], ...]) -> bool:
    return len(left) == len(right) and all(
        set(a) == set(b)
        and all(np.array_equal(np.asarray(a[key]), np.asarray(b[key])) for key in a)
        for a, b in zip(left, right, strict=True)
    )


def _pairwise_distances(positions: np.ndarray) -> list[float]:
    return [
        float(np.linalg.norm(positions[i] - positions[j]))
        for i, j in itertools.combinations(range(4), 2)
    ]


def _pairwise_cosines(actions: np.ndarray) -> list[float | None]:
    result: list[float | None] = []
    for i, j in itertools.combinations(range(4), 2):
        left_norm = float(np.linalg.norm(actions[i]))
        right_norm = float(np.linalg.norm(actions[j]))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            result.append(None)
        else:
            result.append(
                float(np.clip(np.dot(actions[i], actions[j]) / (left_norm * right_norm), -1.0, 1.0))
            )
    return result


def _valid_set_jaccards(valid_ids: tuple[np.ndarray, ...]) -> list[float | None]:
    result: list[float | None] = []
    sets = [set(map(int, ids)) for ids in valid_ids]
    for i, j in itertools.combinations(range(4), 2):
        union = sets[i] | sets[j]
        result.append(float(len(sets[i] & sets[j]) / len(union)) if union else None)
    return result


def _action_implied_targets(
    actions: np.ndarray,
    collector_positions: np.ndarray,
    particle_positions: np.ndarray,
    free_mask: np.ndarray,
) -> list[int]:
    """Return a deterministic action-direction proxy, not a controller target.

    For each nonzero action, choose the free particle with greatest directional
    cosine from the collector; distance and stable particle ID break ties.
    """
    free_ids = np.flatnonzero(free_mask)
    targets: list[int] = []
    for collector, action in zip(collector_positions, actions, strict=True):
        action_norm = float(np.linalg.norm(action))
        if action_norm <= 1e-12 or free_ids.size == 0:
            targets.append(-1)
            continue
        relative = particle_positions[free_ids] - collector
        distances = np.linalg.norm(relative, axis=1)
        cosines = np.divide(
            relative @ (action / action_norm),
            distances,
            out=np.full_like(distances, -np.inf),
            where=distances > 1e-12,
        )
        order = np.lexsort((free_ids, distances, -cosines))
        targets.append(int(free_ids[order[0]]))
    return targets


def _pairing_provenance(
    first: ParticleCollectorEnv, second: ParticleCollectorEnv, *, scenario_seed: int
) -> dict[str, str]:
    assert first.particle_positions is not None and second.particle_positions is not None
    assert first.collector_positions is not None and second.collector_positions is not None
    assert first._noise is not None and second._noise is not None
    if first.scenario_seed != scenario_seed or second.scenario_seed != scenario_seed:
        raise RuntimeError("pairing mismatch: scenario seed provenance")
    if not np.array_equal(first.particle_positions, second.particle_positions):
        raise RuntimeError("pairing mismatch: initial particle positions")
    if not np.array_equal(first.collector_positions, second.collector_positions):
        raise RuntimeError("pairing mismatch: initial collector positions")
    if not np.array_equal(first._noise, second._noise):
        raise RuntimeError("pairing mismatch: Brownian tensor")
    if (
        first.field_seed != second.field_seed
        or first.field_realization_sha256 is None
        or first.field_realization_sha256 != second.field_realization_sha256
    ):
        raise RuntimeError("pairing mismatch: field realization")
    if first.tie_scheme != EVENT_KEYED_TIE_SCHEME or second.tie_scheme != EVENT_KEYED_TIE_SCHEME:
        raise RuntimeError("pairing mismatch: event-keyed tie definition")
    initial = {
        "particle_positions_sha256": _array_sha256(first.particle_positions),
        "collector_positions_sha256": _array_sha256(first.collector_positions),
    }
    tie_key = {
        "scheme": EVENT_KEYED_TIE_SCHEME,
        "scenario_seed": scenario_seed,
        "event_key": ["scenario_seed", "one_based_step_index", "stable_particle_id"],
        "candidate_order": "sorted_stable_collector_ids",
    }
    return {
        "initial_state_sha256": sha256_json(initial),
        "brownian_tensor_sha256": _array_sha256(first._noise),
        "field_realization_sha256": first.field_realization_sha256,
        "policy_randomness_sha256": sha256_json(
            {"policy": "capacity_matched_velocity_controller", "randomness": "none"}
        ),
        "tie_key_provenance_sha256": sha256_json(tie_key),
    }


def _diagnostic_record(
    *,
    run_id: str,
    arm: Arm,
    mode: AggregationMode,
    scenario_seed: int,
    env: ParticleCollectorEnv,
    decision: CapacityMatchedDecision,
    source_state_sha256: str,
    decision_step: int,
    snapshot: Any,
    source_particle_positions: np.ndarray,
    source_capture_free: np.ndarray,
    outcome_collector_positions: np.ndarray,
    info: Mapping[str, Any],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    assert env.field_seed is not None
    assert env._frozen_field is not None
    correlation_length = float(env.config.field_kwargs["correlation_length"])
    spacing = float(np.sqrt(np.prod(env.config.arena_size) / 4.0))
    reasons: list[str] = []
    if decision_step == 0:
        reasons.append("initial_no_history")
    if np.any(decision.own_empty):
        reasons.append("empty_summary")
    if np.any(snapshot.reflected_valid_counts):
        reasons.append("reflection")
    if np.any(decision.clipped_components):
        reasons.append("clipping")
    if np.any(decision.fallback_used):
        reasons.append("fallback")
    eligible = not reasons
    latent: list[list[float]] | None = None
    noise: list[list[float]] | None = None
    risk: float | None = None
    if eligible:
        latent_array, noise_array = sensed_summary_covariances(
            snapshot.particle_source_positions,
            snapshot.valid_mask,
            lambda displacement: env.config.signal_strength**2
            * env._frozen_field.component_covariance(displacement),
            apparent_velocity_noise_variance=(
                env.config.diffusion_sigma**2 / env.config.dt
            ),
        )
        latent = latent_array.tolist()
        noise = noise_array.tolist()
        risk = global_minus_local_estimator_risk(latent_array, noise_array)

    source_collectors = np.asarray(snapshot.collector_positions)
    displacement = outcome_collector_positions - source_collectors
    adjacency = (
        np.eye(4, dtype=np.bool_)
        if mode == "independent"
        else np.ones((4, 4), dtype=np.bool_)
    )
    estimator_error = np.sum(
        (decision.received_messages[:, :2] - snapshot.true_local_velocities) ** 2,
        axis=1,
    )
    implied_targets = _action_implied_targets(
        decision.actions,
        source_collectors,
        source_particle_positions,
        source_capture_free,
    )
    target_pairs = list(itertools.combinations(implied_targets, 2))
    target_overlap = sum(
        left >= 0 and left == right for left, right in target_pairs
    ) / len(target_pairs)
    return {
        "schema_version": 2,
        "run_id": run_id,
        "arm": arm,
        "scenario_seed": scenario_seed,
        "field_seed": env.field_seed,
        "correlation_length": correlation_length,
        "eta": correlation_length / spacing,
        "decision_step": decision_step,
        "outcome_step": int(info["step"]),
        "aggregation_mode": mode,
        "adjacency": adjacency.tolist(),
        "source_state_sha256": source_state_sha256,
        "raw_summaries": decision.raw_summaries.tolist(),
        "sent_messages": decision.sent_messages.tolist(),
        "received_messages": decision.received_messages.tolist(),
        "valid_particle_ids": [ids.tolist() for ids in snapshot.valid_particle_ids],
        "valid_particle_counts": [int(ids.size) for ids in snapshot.valid_particle_ids],
        "valid_source_positions": [positions.tolist() for positions in snapshot.valid_particle_source_positions],
        "true_local_velocities": snapshot.true_local_velocities.tolist(),
        "conditional_field_covariance": latent,
        "conditional_noise_covariance": noise,
        "global_minus_local_estimator_risk": risk,
        "analytic_eligible": eligible,
        "analytic_ineligibility_reasons": reasons,
        "estimator_squared_errors": estimator_error.tolist(),
        "clipped_components": decision.clipped_components.tolist(),
        "own_empty": decision.own_empty.tolist(),
        "fallback_used": decision.fallback_used.tolist(),
        "cross_agent_rescue": decision.cross_agent_rescue.tolist(),
        "zero_direction_cancellation": decision.zero_direction_cancellation.tolist(),
        "reflected_valid_counts": snapshot.reflected_valid_counts.tolist(),
        "collector_reflected": snapshot.collector_reflected.tolist(),
        "source_collector_positions": source_collectors.tolist(),
        "pairwise_collector_distances": _pairwise_distances(source_collectors),
        "outcome_collector_positions": outcome_collector_positions.tolist(),
        "realized_collector_displacements": displacement.tolist(),
        "outcome_pairwise_collector_distances": _pairwise_distances(outcome_collector_positions),
        "outcome_particle_reflection_count": int(info["particle_reflection_count"]),
        "outcome_collector_reflected": list(info["collector_reflected"]),
        "pairwise_valid_set_jaccards": _valid_set_jaccards(snapshot.valid_particle_ids),
        "collector_actions": decision.actions.tolist(),
        "action_implied_target_ids": implied_targets,
        "pursuit_target_ids": implied_targets,
        "pursuit_overlap_fraction": float(target_overlap),
        "pairwise_action_cosines": _pairwise_cosines(decision.actions),
        "transition_unique_captures": len(info["captures"]),
        "capture_events": [
            {"particle_id": int(particle_id), "collector_id": int(collector_id)}
            for particle_id, collector_id in info["captures"]
        ],
        "tie_provenance": [
            {
                "step_index": int(row["step_index"]),
                "particle_id": int(row["particle_id"]),
                "candidate_collector_ids": [
                    int(value) for value in row["candidate_collector_ids"]
                ],
                "chosen_collector_id": int(row["chosen_collector_id"]),
            }
            for row in info["tie_provenance"]
        ],
        "unique_yield_so_far": int(info["captured_total"]),
        "provenance": dict(provenance),
    }


def _count_rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("rate counts must satisfy 0 <= numerator <= denominator")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": (float(numerator / denominator) if denominator else None),
    }


def _frozen_endpoint_status(*, dt: float, horizon: int) -> str:
    matched = any(
        math.isclose(dt, frozen_dt, rel_tol=0.0, abs_tol=1e-12)
        and horizon == frozen_horizon
        for frozen_dt, frozen_horizon in FROZEN_TIMESTEP_HORIZONS.items()
    )
    return (
        "frozen_T_1.34_matched_physical_endpoint"
        if matched
        else "frozen_T_1.34_current_run_nonendpoint_diagnostic"
    )


def evaluate_arm_eta_diagnostic_gate(
    arm_rates: Mapping[Arm, Mapping[str, float]],
    individual_seed_arm_analytic_rates: Mapping[Arm, list[float]],
) -> dict[str, Any]:
    """Evaluate the frozen nonlinear-data gate after arm-by-eta aggregation.

    Rescue and zero-direction cancellation are intentionally absent: they are
    treatment mechanisms, not invalidity conditions.  Inputs are rates in
    ``[0, 1]``.  The absolute arm-rate gap is also on this fraction scale, so
    the frozen ``0.10`` bound is exactly 10 percentage points.
    """
    required = {
        "analytic_eligibility_rate",
        "clipped_velocity_component_rate",
        "own_empty_agent_row_rate",
        "fallback_agent_row_rate",
        "reflected_valid_slot_rate",
        "collector_reflected_action_transition_rate",
    }
    if set(arm_rates) != set(ARMS) or set(individual_seed_arm_analytic_rates) != set(ARMS):
        raise ValueError("both self_only and all_to_all arms are required")
    for arm in ARMS:
        missing = required - set(arm_rates[arm])
        if missing:
            raise ValueError(f"missing {arm} diagnostic rates: {sorted(missing)}")
        if not individual_seed_arm_analytic_rates[arm]:
            raise ValueError(f"{arm} requires at least one individual seed-arm rate")
        values = [float(arm_rates[arm][key]) for key in required]
        values.extend(float(value) for value in individual_seed_arm_analytic_rates[arm])
        if not values or any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("all diagnostic rates must be finite values in [0, 1]")

    thresholds = FROZEN_DIAGNOSTIC_THRESHOLDS
    components: dict[str, bool] = {}
    for arm in ARMS:
        rates = arm_rates[arm]
        components[f"{arm}_analytic_eligibility"] = (
            rates["analytic_eligibility_rate"]
            >= thresholds["minimum_arm_eta_analytic_eligibility_rate"]
        )
        components[f"{arm}_individual_seed_eligibility"] = all(
            value >= thresholds["minimum_individual_seed_arm_analytic_eligibility_rate"]
            for value in individual_seed_arm_analytic_rates[arm]
        )
        for rate_name, threshold_name in (
            ("clipped_velocity_component_rate", "maximum_clipped_velocity_component_rate"),
            ("own_empty_agent_row_rate", "maximum_own_empty_agent_row_rate"),
            ("fallback_agent_row_rate", "maximum_fallback_agent_row_rate"),
            ("reflected_valid_slot_rate", "maximum_reflected_valid_slot_rate"),
            (
                "collector_reflected_action_transition_rate",
                "maximum_collector_reflected_action_transition_rate",
            ),
        ):
            components[f"{arm}_{rate_name}"] = rates[rate_name] <= thresholds[threshold_name]

    arm_rate_difference = abs(
        arm_rates["self_only"]["analytic_eligibility_rate"]
        - arm_rates["all_to_all"]["analytic_eligibility_rate"]
    )
    components["analytic_eligibility_arm_gap"] = (
        arm_rate_difference
        <= thresholds["maximum_absolute_fraction_difference"] + 1e-12
    )
    return {
        "gate_passed": all(components.values()),
        "components": components,
        "analytic_eligibility_absolute_rate_difference": arm_rate_difference,
        "analytic_eligibility_arm_gap_percentage_points": 100.0 * arm_rate_difference,
    }


def _arm_summary(
    records: list[dict[str, Any]], *, nearest_particles_k: int
) -> dict[str, Any]:
    post_history = [row for row in records if row["decision_step"] >= 1]
    rows = len(post_history)
    agent_rows = 4 * rows
    velocity_components = 8 * rows
    valid_slots = sum(sum(row["valid_particle_counts"]) for row in post_history)
    action_transitions = 4 * rows
    analytic_eligible = sum(bool(row["analytic_eligible"]) for row in post_history)
    clipped = sum(
        sum(sum(agent) for agent in row["clipped_components"])
        for row in post_history
    )
    own_empty = sum(sum(row["own_empty"]) for row in post_history)
    fallback = sum(sum(row["fallback_used"]) for row in post_history)
    reflected_valid = sum(
        sum(row["reflected_valid_counts"]) for row in post_history
    )
    collector_reflected = sum(
        sum(row["outcome_collector_reflected"]) for row in post_history
    )
    rescue = sum(sum(row["cross_agent_rescue"]) for row in post_history)
    cancellation = sum(
        sum(row["zero_direction_cancellation"]) for row in post_history
    )
    nonfallback_agent_rows = agent_rows - fallback
    return {
        "steps": len(records),
        "captured_total": int(records[-1]["unique_yield_so_far"]),
        "agent_steps": 4 * len(records),
        "message_components": 12 * len(records),
        "particle_transition_slots": 4 * nearest_particles_k * len(records),
        "valid_slots": sum(sum(row["valid_particle_counts"]) for row in records),
        "clipped_components": sum(
            sum(sum(agent) for agent in row["clipped_components"]) for row in records
        ),
        "reflected_valid_slots": sum(sum(row["reflected_valid_counts"]) for row in records),
        "outcome_particle_reflections": sum(
            row["outcome_particle_reflection_count"] for row in records
        ),
        "own_empty": sum(sum(row["own_empty"]) for row in records),
        "fallback_used": sum(sum(row["fallback_used"]) for row in records),
        "cross_agent_rescue": sum(sum(row["cross_agent_rescue"]) for row in records),
        "zero_direction_cancellation": sum(sum(row["zero_direction_cancellation"]) for row in records),
        "collector_reflections": sum(
            sum(row["outcome_collector_reflected"]) for row in records
        ),
        "analytic_eligible_rows": sum(bool(row["analytic_eligible"]) for row in records),
        "pairwise_geometry_rows": 6 * len(records),
        "post_history_diagnostics": {
            "rows": rows,
            "agent_rows": agent_rows,
            "velocity_components": velocity_components,
            "valid_slots": valid_slots,
            "action_transitions": action_transitions,
            "analytic_eligibility": _count_rate(analytic_eligible, rows),
            "clipped_velocity_components": _count_rate(clipped, velocity_components),
            "own_empty_agent_rows": _count_rate(own_empty, agent_rows),
            "fallback_agent_rows": _count_rate(fallback, agent_rows),
            "reflected_valid_slots": _count_rate(reflected_valid, valid_slots),
            "collector_reflected_action_transitions": _count_rate(
                collector_reflected, action_transitions
            ),
            "action_displacement_eligible_transitions": _count_rate(
                action_transitions - collector_reflected,
                action_transitions,
            ),
            "cross_agent_rescue_own_empty_rows": _count_rate(rescue, own_empty),
            "zero_direction_cancellation_nonfallback_agent_rows": _count_rate(
                cancellation, nonfallback_agent_rows
            ),
        },
    }


def run_aggregation_pair(
    config: AggregationPairExperimentConfig, output_dir: str | Path
) -> AggregationPairRunResult:
    frozen = {
        "experiment_id": config.experiment_id,
        "scenario_seed": config.scenario_seed,
        "environment": asdict(config.environment),
        "arm_modes": config.arm_modes,
        "repository_commit_sha": config.repository.commit_sha,
    }
    config_sha256 = sha256_json(frozen)
    run_id = _safe_slug(
        f"{config.experiment_id}-aggregation-seed{config.scenario_seed}-"
        f"{config_sha256[:12]}-{config.repository.commit_sha[:12]}"
    )
    filenames = (
        f"{run_id}.self_only.messages.jsonl",
        f"{run_id}.all_to_all.messages.jsonl",
        f"{run_id}.summary.json",
        f"{run_id}.manifest.json",
    )
    directory = Path(output_dir)
    collisions = [directory / name for name in filenames if (directory / name).exists()]
    if collisions:
        raise FileExistsError(f"immutable artifact already exists: {collisions[0]}")

    environments = {arm: ParticleCollectorEnv(config.environment) for arm in ARMS}
    observations: dict[Arm, tuple[Any, ...]] = {}
    for arm in ARMS:
        observations[arm], _ = environments[arm].reset(seed=config.scenario_seed)
    if not _observations_equal(observations["self_only"], observations["all_to_all"]):
        raise RuntimeError("pairing mismatch: reset observations")
    provenance = _pairing_provenance(
        environments["self_only"],
        environments["all_to_all"],
        scenario_seed=config.scenario_seed,
    )

    records: dict[Arm, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    done: dict[Arm, bool] = {arm: False for arm in ARMS}
    equal_source_checks = 0
    equal_sent_checks = 0
    for decision_step in range(config.environment.horizon):
        state_hashes = {
            arm: _state_sha256(environments[arm]) for arm in ARMS if not done[arm]
        }
        decisions: dict[Arm, CapacityMatchedDecision] = {}
        snapshots: dict[Arm, Any] = {}
        for arm in ARMS:
            if done[arm]:
                continue
            decisions[arm] = capacity_matched_velocity_decision(
                observations[arm],
                shared=config.arm_modes[arm] == "all_to_all",
            )
            snapshots[arm] = environments[arm].local_flow_diagnostic_snapshot()
        if set(decisions) == set(ARMS) and state_hashes["self_only"] == state_hashes["all_to_all"]:
            equal_source_checks += 1
            if not np.array_equal(
                decisions["self_only"].sent_messages,
                decisions["all_to_all"].sent_messages,
            ):
                raise RuntimeError("causal contrast failed: equal source states emitted different messages")
            equal_sent_checks += 1

        step_results: dict[Arm, tuple[Any, ...]] = {}
        for arm in ARMS:
            if done[arm]:
                continue
            env = environments[arm]
            assert env.particle_positions is not None
            assert env.capture_state is not None
            source_particles = env.particle_positions.copy()
            source_free = (env.capture_state.owner < 0).copy()
            result = env.step(decisions[arm].actions)
            next_observations, _, terminated, truncated, info = result
            assert env.collector_positions is not None
            row = _diagnostic_record(
                run_id=run_id,
                arm=arm,
                mode=config.arm_modes[arm],
                scenario_seed=config.scenario_seed,
                env=env,
                decision=decisions[arm],
                source_state_sha256=state_hashes[arm],
                decision_step=decision_step,
                snapshot=snapshots[arm],
                source_particle_positions=source_particles,
                source_capture_free=source_free,
                outcome_collector_positions=env.collector_positions.copy(),
                info=info,
                provenance=provenance,
            )
            records[arm].append(row)
            observations[arm] = next_observations
            done[arm] = terminated or truncated
            step_results[arm] = result

        if len(set(config.arm_modes.values())) == 1 and set(step_results) == set(ARMS):
            left_result = step_results["self_only"]
            right_result = step_results["all_to_all"]
            if (
                _state_sha256(environments["self_only"])
                != _state_sha256(environments["all_to_all"])
                or not _observations_equal(left_result[0], right_result[0])
                or not np.array_equal(left_result[1], right_result[1])
                or left_result[2:] != right_result[2:]
            ):
                raise RuntimeError(
                    f"zero-intervention state/event identity failed at step {decision_step}"
                )
            left = records["self_only"][-1]
            right = records["all_to_all"][-1]
            ignored = {"arm"}
            if {k: v for k, v in left.items() if k not in ignored} != {
                k: v for k, v in right.items() if k not in ignored
            }:
                raise RuntimeError(f"zero-intervention identity failed at step {decision_step}")
        if all(done.values()):
            break

    correlation_length = float(config.environment.field_kwargs["correlation_length"])
    spacing = float(np.sqrt(np.prod(config.environment.arena_size) / 4.0))
    summary: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "scenario_seed": config.scenario_seed,
        "correlation_length": correlation_length,
        "eta": correlation_length / spacing,
        "horizon": config.environment.horizon,
        "frozen_physical_endpoint_seconds": FROZEN_PHYSICAL_ENDPOINT_SECONDS,
        "frozen_endpoint_decision_step_semantics": "decision_steps_0_through_66_produce_outcomes_1_through_67;_outcome_68_excluded",
        "arm_modes": dict(config.arm_modes),
        "self_only": _arm_summary(
            records["self_only"],
            nearest_particles_k=config.environment.nearest_particles_k,
        ),
        "all_to_all": _arm_summary(
            records["all_to_all"],
            nearest_particles_k=config.environment.nearest_particles_k,
        ),
        "matched_streams": provenance,
        "equal_source_state_checks": equal_source_checks,
        "equal_sent_message_checks": equal_sent_checks,
        "zero_intervention_identity_checked": len(set(config.arm_modes.values())) == 1,
        "endpoint_status": _frozen_endpoint_status(
            dt=config.environment.dt,
            horizon=config.environment.horizon,
        ),
    }

    schemas = _schema_directory()
    diagnostic_schema = load_schema(schemas / "message_diagnostic.schema.json")
    summary_schema = load_schema(schemas / "aggregation_pair_summary.schema.json")
    manifest_schema = load_schema(schemas / "aggregation_run_manifest.schema.json")
    self_artifact = prepare_jsonl(filenames[0], records["self_only"], diagnostic_schema)
    all_artifact = prepare_jsonl(filenames[1], records["all_to_all"], diagnostic_schema)
    summary_artifact = prepare_json(filenames[2], summary, summary_schema)
    data_artifacts = (self_artifact, all_artifact, summary_artifact)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "created_at_utc": config.created_at_utc,
        "repository": asdict(config.repository),
        "environment": {"config_path": config.config_path, "config_sha256": config_sha256},
        "pairing": {
            "scenario_seed": config.scenario_seed,
            "arms": list(ARMS),
            "pre_generated_noise": True,
            "shared_initial_state": True,
            "same_nonzero_field": True,
            **provenance,
        },
        "intervention": {
            "self_only_mode": config.arm_modes["self_only"],
            "all_to_all_mode": config.arm_modes["all_to_all"],
            "same_encoder": True,
            "same_decoder": True,
            "same_message_dimension": 3,
            "teammates_disabled": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "command": config.run_command,
        },
        "artifacts": [
            {"path": artifact.filename, "sha256": artifact.sha256, "media_type": artifact.media_type, "bytes": len(artifact.payload)}
            for artifact in data_artifacts
        ],
    }
    manifest_artifact: PreparedArtifact = prepare_json(filenames[3], manifest, manifest_schema)
    written = write_immutable_artifacts(directory, (*data_artifacts, manifest_artifact))
    return AggregationPairRunResult(
        run_id=run_id,
        summary=summary,
        manifest=manifest,
        paths=AggregationPairArtifactPaths(
            self_only_diagnostics=written[0],
            all_to_all_diagnostics=written[1],
            summary=written[2],
            manifest=written[3],
        ),
    )


def _environment_from_mapping(value: Mapping[str, Any]) -> ParticleEnvConfig:
    expected = set(ParticleEnvConfig.__dataclass_fields__)
    extras = set(value) - expected
    if extras:
        raise ValueError(f"unknown environment keys: {sorted(extras)}")
    converted = dict(value)
    if "arena_size" in converted:
        converted["arena_size"] = tuple(converted["arena_size"])
    return ParticleEnvConfig(**converted)


def load_aggregation_pair_config(path: str | Path) -> AggregationPairExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("aggregation config root must be a mapping")
    required = {
        "experiment_id", "scenario_seed", "environment", "repository",
        "created_at_utc", "run_command", "arm_modes",
    }
    if set(raw) != required:
        raise ValueError(
            f"config keys invalid; missing={sorted(required - set(raw))}, "
            f"extra={sorted(set(raw) - required)}"
        )
    if not isinstance(raw["environment"], dict) or not isinstance(raw["repository"], dict):
        raise ValueError("environment and repository must be mappings")
    return AggregationPairExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        scenario_seed=int(raw["scenario_seed"]),
        environment=_environment_from_mapping(raw["environment"]),
        repository=RepositoryProvenance(**raw["repository"]),
        created_at_utc=str(raw["created_at_utc"]),
        config_path=config_path.as_posix(),
        run_command=str(raw["run_command"]),
        arm_modes=dict(raw["arm_modes"]),
    )


def run_aggregation_config_file(
    path: str | Path, output_dir: str | Path
) -> AggregationPairRunResult:
    return run_aggregation_pair(load_aggregation_pair_config(path), output_dir)
