"""Local collector observations with an explicit information boundary."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


LocalObservation: TypeAlias = dict[str, NDArray[np.float64] | NDArray[np.bool_]]


def _nearest_visible(
    relative: NDArray[np.float64],
    ids: NDArray[np.int64],
    *,
    sensing_radius: float,
    limit: int,
) -> NDArray[np.int64]:
    distances = np.linalg.norm(relative, axis=1)
    visible = distances <= sensing_radius
    visible_ids = ids[visible]
    visible_distances = distances[visible]
    order = np.lexsort((visible_ids, visible_distances))
    return visible_ids[order[:limit]]


def selected_local_particle_ids(
    particle_positions: ArrayLike,
    collector_positions: ArrayLike,
    free_particle_mask: ArrayLike,
    *,
    sensing_radius: float,
    nearest_particles_k: int,
) -> tuple[NDArray[np.int64], ...]:
    """Return the stable IDs selected for each local observation.

    This is diagnostic metadata, not part of :data:`LocalObservation`.  Keeping
    selection in one helper prevents the runner from reconstructing a subtly
    different top-K set while preserving the agents' original information
    boundary.
    """
    particles = np.asarray(particle_positions, dtype=np.float64)
    collectors = np.asarray(collector_positions, dtype=np.float64)
    free = np.asarray(free_particle_mask, dtype=np.bool_)
    if particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError("particle_positions must have shape (N, 2)")
    if collectors.ndim != 2 or collectors.shape[1] != 2:
        raise ValueError("collector_positions must have shape (M, 2)")
    if free.shape != (particles.shape[0],):
        raise ValueError("free_particle_mask must have shape (N,)")
    if not np.isfinite(sensing_radius) or sensing_radius <= 0:
        raise ValueError("sensing_radius must be finite and positive")
    if nearest_particles_k <= 0:
        raise ValueError("nearest_particles_k must be positive")

    free_ids = np.flatnonzero(free).astype(np.int64)
    return tuple(
        _nearest_visible(
            particles[free_ids] - collector,
            free_ids,
            sensing_radius=sensing_radius,
            limit=nearest_particles_k,
        )
        for collector in collectors
    )


def build_local_observations(
    particle_positions: ArrayLike,
    collector_positions: ArrayLike,
    free_particle_mask: ArrayLike,
    *,
    arena_size: tuple[float, float],
    sensing_radius: float,
    nearest_particles_k: int,
    particle_velocities: ArrayLike | None = None,
    velocity_valid_mask: ArrayLike | None = None,
    dt: float = 1.0,
    include_particle_velocity: bool = True,
    include_teammates: bool = True,
) -> tuple[LocalObservation, ...]:
    """Build observations only from current, locally visible physical state.

    The API deliberately has no latent-field parameter, global summary, random
    generator, or future-noise argument. ``particles`` has K rows and five
    columns: relative x/r, relative y/r, causal finite-difference apparent
    velocity x and y, and radial distance/r. Velocity-derived columns
    are emitted only if that particle was visible to that collector in both
    consecutive observations. Teammate absolute positions are the explicitly
    allowed shared channel, represented relative to self, normalized by arena
    size, and ordered by collector ID.
    """
    particles = np.asarray(particle_positions, dtype=np.float64)
    collectors = np.asarray(collector_positions, dtype=np.float64)
    free = np.asarray(free_particle_mask, dtype=np.bool_)
    if particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError("particle_positions must have shape (N, 2)")
    if collectors.ndim != 2 or collectors.shape[1] != 2:
        raise ValueError("collector_positions must have shape (M, 2)")
    if free.shape != (particles.shape[0],):
        raise ValueError("free_particle_mask must have shape (N,)")
    arena = np.asarray(arena_size, dtype=np.float64)
    if arena.shape != (2,) or not np.all(np.isfinite(arena)) or np.any(arena <= 0):
        raise ValueError("arena_size must contain two finite positive values")
    if not np.isfinite(sensing_radius) or sensing_radius <= 0:
        raise ValueError("sensing_radius must be finite and positive")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if nearest_particles_k <= 0:
        raise ValueError("nearest_particles_k must be positive")

    if particle_velocities is None:
        velocities = np.zeros_like(particles)
    else:
        velocities = np.asarray(particle_velocities, dtype=np.float64)
        if velocities.shape != particles.shape or not np.all(np.isfinite(velocities)):
            raise ValueError("particle_velocities must be finite with shape (N, 2)")
    if velocity_valid_mask is None:
        velocity_valid = np.zeros(
            (collectors.shape[0], particles.shape[0]), dtype=np.bool_
        )
    else:
        velocity_valid = np.asarray(velocity_valid_mask, dtype=np.bool_)
        if velocity_valid.shape != (collectors.shape[0], particles.shape[0]):
            raise ValueError("velocity_valid_mask must have shape (M, N)")

    selected_by_collector = selected_local_particle_ids(
        particles,
        collectors,
        free,
        sensing_radius=sensing_radius,
        nearest_particles_k=nearest_particles_k,
    )
    observations: list[LocalObservation] = []
    for collector_id, collector in enumerate(collectors):
        selected = selected_by_collector[collector_id]
        particle_slots = np.zeros((nearest_particles_k, 5), dtype=np.float64)
        particle_mask = np.zeros(nearest_particles_k, dtype=np.bool_)
        velocity_slot_mask = np.zeros(nearest_particles_k, dtype=np.bool_)
        count = selected.size
        if count:
            selected_relative = particles[selected] - collector
            particle_slots[:count, :2] = selected_relative / sensing_radius
            valid = velocity_valid[collector_id, selected]
            particle_slots[:count, 2:4] = velocities[selected] * valid[:, None]
            particle_slots[:count, 4] = (
                np.linalg.norm(selected_relative, axis=1) / sensing_radius
            )
            particle_mask[:count] = True
            velocity_slot_mask[:count] = valid

        observation: LocalObservation = {
            "self_position": collector / arena,
            "particles": particle_slots,
            "particle_mask": particle_mask,
            "velocity_valid_mask": velocity_slot_mask,
        }
        if not include_particle_velocity:
            observation["particles"][:, 2:4] = 0.0
            observation["velocity_valid_mask"][:] = False
        if include_teammates:
            teammate_ids = np.delete(np.arange(collectors.shape[0]), collector_id)
            observation["teammate_relative_positions"] = (
                collectors[teammate_ids] - collector
            ) / arena
        observations.append(observation)

    return tuple(observations)
