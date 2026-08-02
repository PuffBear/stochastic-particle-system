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


def bounded_team_velocity_summary_v2(
    observations: tuple[LocalObservation, ...], *, leave_out_agent: int | None = None
) -> NDArray[np.float64]:
    """Count-weighted team mean — the Proposition 2 sufficient statistic.

    Weights each agent's local mean by its validity fraction (proportional to
    the number of valid velocity observations it contributed).  Equal-weight
    averaging is a special case when all agents observe the same number of
    valid particles.  The returned f_valid slot is the unweighted mean of
    per-agent fractions, consistent with the original 3-slot message format.
    """
    summaries = np.stack([local_velocity_summary(obs) for obs in observations])
    if leave_out_agent is not None:
        if not 0 <= leave_out_agent < len(observations):
            raise ValueError("leave_out_agent is out of range")
        summaries = np.delete(summaries, leave_out_agent, axis=0)
    if summaries.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    weights = summaries[:, 2]          # validity fractions ∝ observation counts
    total = float(weights.sum())
    if total > 0.0:
        vel = np.average(summaries[:, :2], weights=weights, axis=0)
    else:
        vel = np.zeros(2, dtype=np.float64)
    frac = float(np.mean(weights))
    return np.clip(np.array([vel[0], vel[1], frac], dtype=np.float64), -1.0, 1.0)


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


def capacity_matched_velocity_controller_v2(
    observations: tuple[LocalObservation, ...], *, shared: bool
) -> NDArray[np.float64]:
    """Improved controller with count-weighted team mean and field+density blend.

    shared=True improvements over v1:
      1. Count-weighted team mean (Proposition 2 — gives more weight to agents
         with more valid observations, reducing noise from data-sparse agents).
      2. Field+density blend: each agent blends the shared upstream direction
         with its local particle-density signal.  This prevents the correlated
         failure mode in which all agents simultaneously follow a noisy field
         estimate into an unpopulated region.  The blend weight scales with
         the team's mean validity fraction so the mix is data-driven:
           blend_w = min(0.7, 2 · f_valid_team)
         At the canonical f_valid ≈ 0.35 the split is ≈70/30 field/density.

    shared=False: identical to capacity_matched_velocity_controller v1 so that
    the independent baseline remains an unmodified comparison point.
    """
    team = bounded_team_velocity_summary_v2(observations) if shared else None
    actions = np.zeros((len(observations), 2), dtype=np.float64)
    for agent_id, observation in enumerate(observations):
        mask = np.asarray(observation["particle_mask"], dtype=np.bool_)
        slots = np.asarray(observation["particles"], dtype=np.float64)
        if shared:
            assert team is not None
            if team[2] > 0.0:
                field_dir = _unit(-team[:2])
                blend_w = min(0.7, float(team[2]) * 2.0)
                if np.any(mask):
                    density_dir = _unit(np.mean(slots[mask, :2], axis=0))
                    combined = blend_w * field_dir + (1.0 - blend_w) * density_dir
                    norm = float(np.linalg.norm(combined))
                    actions[agent_id] = _unit(combined) if norm > 1e-12 else field_dir
                else:
                    actions[agent_id] = field_dir
            else:
                if np.any(mask):
                    actions[agent_id] = _unit(np.mean(slots[mask, :2], axis=0))
        else:
            message = local_velocity_summary(observation)
            if message[2] > 0.0:
                actions[agent_id] = _unit(-message[:2])
            else:
                if np.any(mask):
                    actions[agent_id] = _unit(np.mean(slots[mask, :2], axis=0))
    return actions


def capacity_matched_velocity_controller_v2_shuffled(
    observations: tuple[LocalObservation, ...],
    *,
    scenario_seed: int,
    step: int,
) -> NDArray[np.float64]:
    """Ablation: replace team message with reproducible random noise same format.

    The velocity direction and validity fraction are drawn from a deterministic
    RNG seeded by (scenario_seed, step). This tests whether the coordination
    gain is due to message *content* (field-correlated direction) or just the
    extra input slots (bandwidth).
    """
    rng = np.random.default_rng([scenario_seed, step, 0xAB1A710])
    fake = np.array(
        [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(0.0, 1.0)],
        dtype=np.float64,
    )
    actions = np.zeros((len(observations), 2), dtype=np.float64)
    for agent_id, observation in enumerate(observations):
        mask = np.asarray(observation["particle_mask"], dtype=np.bool_)
        slots = np.asarray(observation["particles"], dtype=np.float64)
        if fake[2] > 0.0:
            field_dir = _unit(-fake[:2])
            blend_w = min(0.7, float(fake[2]) * 2.0)
            if np.any(mask):
                density_dir = _unit(np.mean(slots[mask, :2], axis=0))
                combined = blend_w * field_dir + (1.0 - blend_w) * density_dir
                norm = float(np.linalg.norm(combined))
                actions[agent_id] = _unit(combined) if norm > 1e-12 else field_dir
            else:
                actions[agent_id] = field_dir
        else:
            if np.any(mask):
                actions[agent_id] = _unit(np.mean(slots[mask, :2], axis=0))
    return actions


def capacity_matched_velocity_controller_v2_leave_self_out(
    observations: tuple[LocalObservation, ...],
) -> NDArray[np.float64]:
    """Ablation: each agent receives team mean computed without its own observations.

    Tests whether each agent is benefiting from *others'* information or merely
    re-routing its own local estimate through the aggregation arithmetic.
    Uses the same field+density blend logic as v2.
    """
    actions = np.zeros((len(observations), 2), dtype=np.float64)
    for agent_id, observation in enumerate(observations):
        mask = np.asarray(observation["particle_mask"], dtype=np.bool_)
        slots = np.asarray(observation["particles"], dtype=np.float64)
        team = bounded_team_velocity_summary_v2(observations, leave_out_agent=agent_id)
        if team[2] > 0.0:
            field_dir = _unit(-team[:2])
            blend_w = min(0.7, float(team[2]) * 2.0)
            if np.any(mask):
                density_dir = _unit(np.mean(slots[mask, :2], axis=0))
                combined = blend_w * field_dir + (1.0 - blend_w) * density_dir
                norm = float(np.linalg.norm(combined))
                actions[agent_id] = _unit(combined) if norm > 1e-12 else field_dir
            else:
                actions[agent_id] = field_dir
        else:
            if np.any(mask):
                actions[agent_id] = _unit(np.mean(slots[mask, :2], axis=0))
    return actions


def _windowed_team_summary(
    history: list[tuple[LocalObservation, ...]],
    current: tuple[LocalObservation, ...],
    L: int,
) -> NDArray[np.float64]:
    """Count-weighted team mean over the last L steps (inclusive of current).

    history[0] is the oldest observation tuple; current is step t.
    Only the last L-1 history entries plus current are used.
    At L >= len(history)+1 this is equivalent to full-history pooling.
    """
    window = (list(history)[-(L - 1):] if L > 1 else []) + [current]
    vx_sum = 0.0
    vy_sum = 0.0
    w_sum = 0.0
    for obs_tuple in window:
        for obs in obs_tuple:
            s = local_velocity_summary(obs)
            w = s[2]
            vx_sum += w * s[0]
            vy_sum += w * s[1]
            w_sum += w
    if w_sum > 0.0:
        vel = np.array([vx_sum / w_sum, vy_sum / w_sum], dtype=np.float64)
    else:
        vel = np.zeros(2, dtype=np.float64)
    frac = float(
        np.mean([local_velocity_summary(obs)[2] for obs in current])
    )
    return np.clip(np.array([vel[0], vel[1], frac], dtype=np.float64), -1.0, 1.0)


def _decay_team_summary(
    history: list[tuple[LocalObservation, ...]],
    current: tuple[LocalObservation, ...],
    L: int,
) -> NDArray[np.float64]:
    """Exponentially-decayed count-weighted team mean with lambda = exp(-1/L).

    Weights observations at lag k with lambda^k. Current step has weight 1.
    history[0] is the oldest; current is step t.
    """
    lam = float(np.exp(-1.0 / max(L, 1)))
    all_steps = list(history) + [current]
    n = len(all_steps)
    vx_sum = 0.0
    vy_sum = 0.0
    w_sum = 0.0
    for lag, obs_tuple in enumerate(reversed(all_steps)):
        decay = lam ** lag
        for obs in obs_tuple:
            s = local_velocity_summary(obs)
            w = decay * s[2]
            vx_sum += w * s[0]
            vy_sum += w * s[1]
            w_sum += w
    if w_sum > 0.0:
        vel = np.array([vx_sum / w_sum, vy_sum / w_sum], dtype=np.float64)
    else:
        vel = np.zeros(2, dtype=np.float64)
    frac = float(
        np.mean([local_velocity_summary(obs)[2] for obs in current])
    )
    return np.clip(np.array([vel[0], vel[1], frac], dtype=np.float64), -1.0, 1.0)


def _apply_field_density_blend(
    team: NDArray[np.float64],
    observation: LocalObservation,
    actions: NDArray[np.float64],
    agent_id: int,
) -> None:
    """In-place action computation using field+density blend (v2 logic)."""
    mask = np.asarray(observation["particle_mask"], dtype=np.bool_)
    slots = np.asarray(observation["particles"], dtype=np.float64)
    if team[2] > 0.0:
        field_dir = _unit(-team[:2])
        blend_w = min(0.7, float(team[2]) * 2.0)
        if np.any(mask):
            density_dir = _unit(np.mean(slots[mask, :2], axis=0))
            combined = blend_w * field_dir + (1.0 - blend_w) * density_dir
            norm = float(np.linalg.norm(combined))
            actions[agent_id] = _unit(combined) if norm > 1e-12 else field_dir
        else:
            actions[agent_id] = field_dir
    else:
        if np.any(mask):
            actions[agent_id] = _unit(np.mean(slots[mask, :2], axis=0))


def capacity_matched_velocity_controller_v2_window(
    observations: tuple[LocalObservation, ...],
    history: list[tuple[LocalObservation, ...]],
    L: int,
    *,
    shared: bool,
) -> NDArray[np.float64]:
    """FR-B4 sliding-window controller: last L steps, count-weighted team mean.

    At L=all (L >= episode length) and omega=0 this reproduces v2 exactly.
    shared=False uses the same window over each agent's own local observations.
    history should be the list of past observation tuples (oldest first).
    """
    actions = np.zeros((len(observations), 2), dtype=np.float64)
    if shared:
        team = _windowed_team_summary(history, observations, L)
        for agent_id, observation in enumerate(observations):
            _apply_field_density_blend(team, observation, actions, agent_id)
    else:
        for agent_id, observation in enumerate(observations):
            solo_history = [(obs_tuple[agent_id],) for obs_tuple in history]
            team = _windowed_team_summary(solo_history, (observation,), L)
            _apply_field_density_blend(team, observation, actions, agent_id)
    return actions


def capacity_matched_velocity_controller_v2_decay(
    observations: tuple[LocalObservation, ...],
    history: list[tuple[LocalObservation, ...]],
    L: int,
    *,
    shared: bool,
) -> NDArray[np.float64]:
    """FR-B4 exponential-decay controller: lambda=exp(-1/L), count-weighted.

    At L=all and omega=0 the decay is negligible and this approaches v2.
    shared=False applies decay over each agent's own local observations.
    history should be the list of past observation tuples (oldest first).
    """
    actions = np.zeros((len(observations), 2), dtype=np.float64)
    if shared:
        team = _decay_team_summary(history, observations, L)
        for agent_id, observation in enumerate(observations):
            _apply_field_density_blend(team, observation, actions, agent_id)
    else:
        for agent_id, observation in enumerate(observations):
            solo_history = [(obs_tuple[agent_id],) for obs_tuple in history]
            team = _decay_team_summary(solo_history, (observation,), L)
            _apply_field_density_blend(team, observation, actions, agent_id)
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
