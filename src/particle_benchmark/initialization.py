"""Deterministic, capture-free episode initialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class InitialState:
    """Positions sampled before an episode begins."""

    particle_positions: NDArray[np.float64]
    collector_positions: NDArray[np.float64]


def collector_lattice(
    collector_count: int, arena_size: tuple[float, float]
) -> NDArray[np.float64]:
    """Place collectors deterministically at cell centres of a compact grid.

    The canonical four-collector case is ordered by collector column then row at
    ``(.25,.25), (.25,.75), (.75,.25), (.75,.75)`` in the unit arena.
    """
    if collector_count <= 0:
        raise ValueError("collector_count must be positive")
    arena = np.asarray(arena_size, dtype=np.float64)
    if arena.shape != (2,) or not np.all(np.isfinite(arena)) or np.any(arena <= 0):
        raise ValueError("arena_size must contain two finite positive values")
    columns = int(np.ceil(np.sqrt(collector_count)))
    rows = int(np.ceil(collector_count / columns))
    x = (np.arange(columns, dtype=np.float64) + 0.5) / columns * arena[0]
    y = (np.arange(rows, dtype=np.float64) + 0.5) / rows * arena[1]
    return np.array([(x_value, y_value) for x_value in x for y_value in y])[
        :collector_count
    ]


def sample_capture_free_initial_state(
    rng: np.random.Generator,
    *,
    particle_count: int,
    collector_count: int,
    arena_size: tuple[float, float],
    exclusion_radius: float,
    clearance: float = 0.0,
    max_attempts_per_particle: int = 10_000,
) -> InitialState:
    """Sample a reproducible reset with no particle in capture range.

    Collectors are placed on a deterministic lattice. Each particle is rejection sampled
    until its distance from every collector is strictly greater than the
    capture exclusion radius plus the requested numerical clearance.
    """
    if particle_count <= 0 or collector_count <= 0:
        raise ValueError("particle_count and collector_count must be positive")
    arena = np.asarray(arena_size, dtype=np.float64)
    if arena.shape != (2,) or not np.all(np.isfinite(arena)) or np.any(arena <= 0):
        raise ValueError("arena_size must contain two finite positive values")
    if (
        not np.isfinite(exclusion_radius)
        or not np.isfinite(clearance)
        or exclusion_radius < 0
        or clearance < 0
    ):
        raise ValueError("exclusion_radius and clearance must be non-negative")
    if max_attempts_per_particle <= 0:
        raise ValueError("max_attempts_per_particle must be positive")

    collectors = collector_lattice(collector_count, arena_size)
    particles = np.empty((particle_count, 2), dtype=np.float64)
    minimum_distance = exclusion_radius + clearance
    for particle_id in range(particle_count):
        for _ in range(max_attempts_per_particle):
            candidate = rng.uniform(0.0, arena, size=2)
            distances = np.linalg.norm(collectors - candidate, axis=1)
            if np.all(distances > minimum_distance):
                particles[particle_id] = candidate
                break
        else:
            raise RuntimeError(
                "could not sample a capture-free reset; reduce density or exclusion radius"
            )

    return InitialState(
        particle_positions=particles,
        collector_positions=collectors,
    )
