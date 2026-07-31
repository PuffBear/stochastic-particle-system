"""Frozen scripted policies for correctness and bounded baseline studies."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dynamics.fields import field_velocity
from .observations import LocalObservation


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a finite unit vector, or zero when the input has no direction."""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros(2, dtype=np.float64)
    return vector / norm


def stationary_policy(collector_count: int) -> NDArray[np.float64]:
    """Return zero action for every collector."""
    if collector_count <= 0:
        raise ValueError("collector_count must be positive")
    return np.zeros((collector_count, 2), dtype=np.float64)


def random_policy(
    collector_count: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sample actions uniformly by direction and within unit-disc area."""
    if collector_count <= 0:
        raise ValueError("collector_count must be positive")
    angle = rng.uniform(0.0, 2.0 * np.pi, size=collector_count)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=collector_count))
    return radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))


def random_action_tensor(
    seed: int, *, horizon: int, collector_count: int
) -> NDArray[np.float64]:
    """Pre-generate a complete random-policy action tensor for matched pairs."""
    if horizon <= 0 or collector_count <= 0:
        raise ValueError("horizon and collector_count must be positive")
    from .seeding import make_streams

    rng = make_streams(seed).policy
    angle = rng.uniform(0.0, 2.0 * np.pi, size=(horizon, collector_count))
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=(horizon, collector_count)))
    return radius[..., None] * np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def privileged_field_policy(
    collector_positions: ArrayLike,
    *,
    field_family: str,
    signal_strength: float,
    field_kwargs: dict[str, object] | None = None,
) -> NDArray[np.float64]:
    """Move upstream against the true field, intentionally using privilege."""
    positions = np.asarray(collector_positions, dtype=np.float64)
    velocity = field_velocity(
        positions,
        field_family,
        signal_strength,
        **(field_kwargs or {}),
    )
    norm = np.linalg.norm(velocity, axis=1, keepdims=True)
    return -np.divide(velocity, norm, out=np.zeros_like(velocity), where=norm > 0)


def density_greedy_policy(
    observations: tuple[LocalObservation, ...],
) -> NDArray[np.float64]:
    """Move each collector toward the centroid of its visible particles.

    This baseline uses relative positions and presence masks only. It never
    consumes apparent velocity, teammate state, or latent field parameters.
    """
    actions = np.zeros((len(observations), 2), dtype=np.float64)
    for collector_id, observation in enumerate(observations):
        mask = np.asarray(observation["particle_mask"], dtype=np.bool_)
        slots = np.asarray(observation["particles"], dtype=np.float64)
        if np.any(mask):
            actions[collector_id] = _unit(np.mean(slots[mask, :2], axis=0))
    return actions


def local_flow_v1_policy(
    observations: tuple[LocalObservation, ...],
) -> NDArray[np.float64]:
    """Frozen local policy: move against mean causally valid particle velocity.

    For each collector independently, average the apparent velocities of slots
    whose particle, velocity, and presence masks are all valid. Move at unit
    normalized speed opposite that mean. If no velocity is valid, remain
    stationary. No density-seeking fallback is used, so the comparison against
    ``density_greedy_policy`` isolates use of local motion information.
    """
    actions = np.zeros((len(observations), 2), dtype=np.float64)
    for collector_id, observation in enumerate(observations):
        present = np.asarray(observation["particle_mask"], dtype=np.bool_)
        velocity_valid = np.asarray(
            observation["velocity_valid_mask"], dtype=np.bool_
        )
        valid = present & velocity_valid
        slots = np.asarray(observation["particles"], dtype=np.float64)
        if np.any(valid):
            actions[collector_id] = _unit(-np.mean(slots[valid, 2:4], axis=0))
    return actions


def coverage_policy(
    observations: tuple[LocalObservation, ...],
    *,
    step: int,
    sweep_period: int = 100,
) -> NDArray[np.float64]:
    """Deterministic lane-coverage control independent of particle evidence."""
    if step < 0 or sweep_period <= 0:
        raise ValueError("step must be non-negative and sweep_period positive")
    count = len(observations)
    if count <= 0:
        raise ValueError("at least one observation is required")
    target_x = 0.95 if (step // sweep_period) % 2 == 0 else 0.05
    actions = np.zeros((count, 2), dtype=np.float64)
    for collector_id, observation in enumerate(observations):
        position = np.asarray(observation["self_position"], dtype=np.float64)
        target = np.array(
            [target_x, (collector_id + 0.5) / count], dtype=np.float64
        )
        actions[collector_id] = _unit(target - position)
    return actions
