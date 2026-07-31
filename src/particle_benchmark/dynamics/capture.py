"""Permanent fixed- and growing-geometry particle capture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class CaptureState:
    """Mutable capture ownership and collector-relative aggregate offsets."""

    owner: NDArray[np.int32]
    offsets: list[list[NDArray[np.float64]]]

    @classmethod
    def empty(cls, particle_count: int, collector_count: int) -> "CaptureState":
        if particle_count < 0 or collector_count <= 0:
            raise ValueError("invalid particle or collector count")
        return cls(
            owner=np.full(particle_count, -1, dtype=np.int32),
            offsets=[[] for _ in range(collector_count)],
        )


def _collector_centres(
    collector_position: NDArray[np.float64],
    offsets: list[NDArray[np.float64]],
    geometry: str,
) -> NDArray[np.float64]:
    if geometry == "fixed" or not offsets:
        return collector_position[None, :]
    if geometry != "growing":
        raise ValueError(f"unknown capture geometry: {geometry}")
    return np.vstack([collector_position, collector_position + np.vstack(offsets)])


def resolve_captures(
    particle_positions: ArrayLike,
    collector_positions: ArrayLike,
    state: CaptureState,
    *,
    geometry: str,
    collector_radius: float,
    particle_radius: float,
    tie_rng: np.random.Generator,
    tie_tolerance: float = 1e-12,
) -> list[tuple[int, int]]:
    """Capture every touching free particle exactly once.

    Ownership goes to the closest collector aggregate. Exact ties are resolved
    by the dedicated tie RNG. New growing-geometry nodes are available only on
    the next call, preventing iteration-order cascades inside one time step.
    """
    particles = np.asarray(particle_positions, dtype=np.float64)
    collectors = np.asarray(collector_positions, dtype=np.float64)
    if particles.ndim != 2 or particles.shape[1] != 2:
        raise ValueError("particle_positions must have shape (N, 2)")
    if collectors.ndim != 2 or collectors.shape[1] != 2:
        raise ValueError("collector_positions must have shape (M, 2)")
    if state.owner.shape != (particles.shape[0],):
        raise ValueError("owner array does not match particle count")
    if len(state.offsets) != collectors.shape[0]:
        raise ValueError("offset list does not match collector count")
    if geometry not in {"fixed", "growing"}:
        raise ValueError(f"unknown capture geometry: {geometry}")
    contact = collector_radius + particle_radius
    if contact < 0 or not np.isfinite(contact):
        raise ValueError("capture radii must be finite and non-negative")

    aggregate_centres = [
        _collector_centres(collectors[i], state.offsets[i], geometry)
        for i in range(collectors.shape[0])
    ]
    events: list[tuple[int, int]] = []
    for particle_id in np.flatnonzero(state.owner < 0):
        minimums = np.array(
            [
                np.min(np.linalg.norm(centres - particles[particle_id], axis=1))
                for centres in aggregate_centres
            ]
        )
        eligible = np.flatnonzero(minimums <= contact)
        if eligible.size == 0:
            continue
        best_distance = np.min(minimums[eligible])
        tied = eligible[
            np.isclose(
                minimums[eligible],
                best_distance,
                atol=tie_tolerance,
                rtol=0.0,
            )
        ]
        collector_id = int(tied[0] if tied.size == 1 else tie_rng.choice(tied))
        state.owner[particle_id] = collector_id
        if geometry == "growing":
            state.offsets[collector_id].append(
                (particles[particle_id] - collectors[collector_id]).copy()
            )
        events.append((int(particle_id), collector_id))
    return events

