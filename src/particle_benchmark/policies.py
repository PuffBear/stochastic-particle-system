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


def full_state_interception_oracle(
    collector_positions: ArrayLike,
    particle_positions: ArrayLike,
    particle_velocities: ArrayLike,
    free_mask: ArrayLike,
    *,
    collector_max_speed: float,
    receding_horizon: float,
) -> NDArray[np.float64]:
    """Action-feasible current-state oracle with no access to future noise.

    Constant-velocity intercept times are solved analytically.  Candidate
    collector/particle pairs are greedily assigned in increasing intercept
    time, with stable IDs breaking ties.  Targets faster than the collector are
    still admissible when geometry permits a finite intercept.  Boundary
    reflections beyond the current step are deliberately not predicted.
    """
    collectors = np.asarray(collector_positions, dtype=np.float64)
    particles = np.asarray(particle_positions, dtype=np.float64)
    velocities = np.asarray(particle_velocities, dtype=np.float64)
    free = np.asarray(free_mask, dtype=np.bool_)
    if collectors.ndim != 2 or collectors.shape[1:] != (2,):
        raise ValueError("collector_positions must have shape (M, 2)")
    if particles.shape != velocities.shape or particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError("particle positions and velocities must have shape (N, 2)")
    if free.shape != (particles.shape[0],):
        raise ValueError("free_mask must have shape (N,)")
    if collector_max_speed <= 0 or receding_horizon <= 0:
        raise ValueError("speed and receding_horizon must be positive")

    candidates: list[tuple[float, int, int, NDArray[np.float64]]] = []
    speed2 = collector_max_speed**2
    for collector_id, collector in enumerate(collectors):
        for particle_id in np.flatnonzero(free):
            relative = particles[particle_id] - collector
            velocity = velocities[particle_id]
            a = float(np.dot(velocity, velocity) - speed2)
            b = float(2.0 * np.dot(relative, velocity))
            c = float(np.dot(relative, relative))
            roots: list[float] = []
            if c <= 1e-24:
                roots = [0.0]
            elif abs(a) <= 1e-14:
                if abs(b) > 1e-14:
                    roots = [-c / b]
            else:
                discriminant = b * b - 4.0 * a * c
                if discriminant >= 0.0:
                    root = float(np.sqrt(discriminant))
                    roots = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
            feasible = [t for t in roots if 0.0 <= t <= receding_horizon]
            if feasible:
                intercept_time = min(feasible)
                point = particles[particle_id] + intercept_time * velocity
                candidates.append((intercept_time, collector_id, int(particle_id), point))

    actions = np.zeros_like(collectors)
    used_collectors: set[int] = set()
    used_particles: set[int] = set()
    for _, collector_id, particle_id, point in sorted(
        candidates, key=lambda row: (row[0], row[1], row[2])
    ):
        if collector_id in used_collectors or particle_id in used_particles:
            continue
        actions[collector_id] = _unit(point - collectors[collector_id])
        used_collectors.add(collector_id)
        used_particles.add(particle_id)
    return actions


def local_velocity_summary(observation: LocalObservation) -> NDArray[np.float64]:
    """Return clipped mean apparent velocity and validity fraction."""
    present = np.asarray(observation["particle_mask"], dtype=np.bool_)
    velocity_valid = np.asarray(observation["velocity_valid_mask"], dtype=np.bool_)
    valid = present & velocity_valid
    slots = np.asarray(observation["particles"], dtype=np.float64)
    velocity = np.mean(slots[valid, 2:4], axis=0) if np.any(valid) else np.zeros(2)
    velocity = np.clip(velocity, -1.0, 1.0)
    fraction = float(np.mean(valid)) if valid.size else 0.0
    return np.array([velocity[0], velocity[1], fraction], dtype=np.float64)


def bounded_team_velocity_summary(
    observations: tuple[LocalObservation, ...], *, leave_out_agent: int | None = None
) -> NDArray[np.float64]:
    """Permutation-invariant three-number bounded communication channel."""
    summaries = np.stack([local_velocity_summary(obs) for obs in observations])
    if leave_out_agent is not None:
        if not 0 <= leave_out_agent < len(observations):
            raise ValueError("leave_out_agent is out of range")
        summaries = np.delete(summaries, leave_out_agent, axis=0)
    if summaries.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    return np.clip(np.mean(summaries, axis=0), -1.0, 1.0)


def capacity_matched_velocity_controller(
    observations: tuple[LocalObservation, ...], *, shared: bool
) -> NDArray[np.float64]:
    """Identical-shape controller using either local or shared three slots."""
    team = bounded_team_velocity_summary(observations) if shared else None
    actions = np.zeros((len(observations), 2), dtype=np.float64)
    for agent_id, observation in enumerate(observations):
        message = team if shared else local_velocity_summary(observation)
        if message[2] > 0.0:
            actions[agent_id] = _unit(-message[:2])
        else:
            # Identical deterministic density fallback in both controllers.
            mask = np.asarray(observation["particle_mask"], dtype=np.bool_)
            slots = np.asarray(observation["particles"], dtype=np.float64)
            if np.any(mask):
                actions[agent_id] = _unit(np.mean(slots[mask, :2], axis=0))
    return actions


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
