"""Permanent fixed- and growing-geometry particle capture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .boundaries import reflect


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


@dataclass(frozen=True)
class ContactEvent:
    """One fixed-geometry swept contact in stable entity-ID coordinates."""

    particle_id: int
    collector_id: int
    time_fraction: float
    closest_distance: float


@dataclass(frozen=True)
class TieDecision:
    """Auditable event-keyed ownership choice for an exact contact tie."""

    step_index: int
    particle_id: int
    candidate_collector_ids: tuple[int, ...]
    chosen_collector_id: int


@dataclass(frozen=True)
class SweptCaptureResult:
    """Swept contacts plus diagnostics for deliberately guarded path pairs."""

    events: tuple[ContactEvent, ...]
    tie_decisions: tuple[TieDecision, ...]
    guarded_reflection_pairs: int


def _first_contact_fraction(
    particle_start: NDArray[np.float64],
    particle_end: NDArray[np.float64],
    collector_start: NDArray[np.float64],
    collector_end: NDArray[np.float64],
    contact_radius: float,
    *,
    tolerance: float,
) -> tuple[float, float] | None:
    """Return earliest chord intersection and closest chord distance."""
    relative_start = particle_start - collector_start
    relative_delta = (
        particle_end - particle_start - (collector_end - collector_start)
    )
    radius_squared = contact_radius * contact_radius
    c = float(np.dot(relative_start, relative_start) - radius_squared)
    quadratic = float(np.dot(relative_delta, relative_delta))
    if c <= tolerance:
        return 0.0, float(np.linalg.norm(relative_start))
    if quadratic <= tolerance:
        return None

    linear = 2.0 * float(np.dot(relative_start, relative_delta))
    discriminant = linear * linear - 4.0 * quadratic * c
    if discriminant < -tolerance:
        return None
    discriminant = max(discriminant, 0.0)
    root = (-linear - np.sqrt(discriminant)) / (2.0 * quadratic)
    if root < -tolerance or root > 1.0 + tolerance:
        return None
    root = float(np.clip(root, 0.0, 1.0))
    closest_fraction = float(
        np.clip(-np.dot(relative_start, relative_delta) / quadratic, 0.0, 1.0)
    )
    closest_distance = float(
        np.linalg.norm(relative_start + closest_fraction * relative_delta)
    )
    return root, closest_distance


def _event_keyed_choice(
    candidates: NDArray[np.int64],
    *,
    event_seed: int,
    step_index: int,
    particle_id: int,
) -> int:
    """Choose from stable IDs without depending on prior RNG consumption."""
    stable = np.sort(candidates)
    sequence = np.random.SeedSequence([event_seed, step_index, particle_id])
    rng = np.random.default_rng(sequence)
    return int(stable[int(rng.integers(0, stable.size))])


def _wall_crossing_fractions(
    start: NDArray[np.float64],
    unfolded_end: NDArray[np.float64],
    arena_size: NDArray[np.float64],
    *,
    tolerance: float,
) -> list[float]:
    """Return all strict interior wall-hit fractions for an unfolded chord."""
    fractions: list[float] = []
    for axis in range(2):
        displacement = float(unfolded_end[axis] - start[axis])
        if abs(displacement) <= tolerance:
            continue
        length = float(arena_size[axis])
        lower = min(float(start[axis]), float(unfolded_end[axis]))
        upper = max(float(start[axis]), float(unfolded_end[axis]))
        first_wall = int(np.ceil(lower / length))
        last_wall = int(np.floor(upper / length))
        for wall_index in range(first_wall, last_wall + 1):
            fraction = (wall_index * length - float(start[axis])) / displacement
            if tolerance < fraction < 1.0 - tolerance:
                fractions.append(float(fraction))
    return fractions


def _merged_breakpoints(values: list[float], *, tolerance: float) -> list[float]:
    """Sort and merge numerically coincident wall/corner events."""
    ordered = sorted([0.0, 1.0, *values])
    merged: list[float] = []
    for value in ordered:
        if not merged or value - merged[-1] > tolerance:
            merged.append(value)
        else:
            merged[-1] = min(merged[-1], value)
    merged[0] = 0.0
    merged[-1] = 1.0
    return merged


def _reflected_position_at(
    start: NDArray[np.float64],
    unfolded_end: NDArray[np.float64],
    fraction: float,
    arena_size: NDArray[np.float64],
) -> NDArray[np.float64]:
    unfolded = start + fraction * (unfolded_end - start)
    return np.asarray(reflect(unfolded, arena_size), dtype=np.float64)


def _piecewise_reflected_first_contact(
    particle_start: NDArray[np.float64],
    particle_unfolded_end: NDArray[np.float64],
    collector_start: NDArray[np.float64],
    collector_unfolded_end: NDArray[np.float64],
    arena_size: NDArray[np.float64],
    contact_radius: float,
    *,
    time_tolerance: float,
    distance_tolerance: float,
) -> tuple[float, float] | None:
    """Solve earliest contact across exact rectangular specular segments."""
    wall_fractions = _wall_crossing_fractions(
        particle_start,
        particle_unfolded_end,
        arena_size,
        tolerance=time_tolerance,
    )
    wall_fractions.extend(
        _wall_crossing_fractions(
            collector_start,
            collector_unfolded_end,
            arena_size,
            tolerance=time_tolerance,
        )
    )
    breakpoints = _merged_breakpoints(wall_fractions, tolerance=time_tolerance)
    for segment_start, segment_end in zip(
        breakpoints[:-1], breakpoints[1:], strict=True
    ):
        particle_segment_start = _reflected_position_at(
            particle_start, particle_unfolded_end, segment_start, arena_size
        )
        particle_segment_end = _reflected_position_at(
            particle_start, particle_unfolded_end, segment_end, arena_size
        )
        collector_segment_start = _reflected_position_at(
            collector_start, collector_unfolded_end, segment_start, arena_size
        )
        collector_segment_end = _reflected_position_at(
            collector_start, collector_unfolded_end, segment_end, arena_size
        )
        local_contact = _first_contact_fraction(
            particle_segment_start,
            particle_segment_end,
            collector_segment_start,
            collector_segment_end,
            contact_radius,
            tolerance=max(time_tolerance, distance_tolerance),
        )
        if local_contact is None:
            continue
        local_fraction, closest_distance = local_contact
        global_fraction = segment_start + local_fraction * (
            segment_end - segment_start
        )
        return float(global_fraction), closest_distance
    return None


def resolve_swept_fixed_captures(
    particle_start_positions: ArrayLike,
    particle_end_positions: ArrayLike,
    collector_start_positions: ArrayLike,
    collector_end_positions: ArrayLike,
    state: CaptureState,
    *,
    collector_radius: float,
    particle_radius: float,
    event_seed: int,
    step_index: int,
    particle_ids: ArrayLike | None = None,
    collector_ids: ArrayLike | None = None,
    particle_reflected: ArrayLike | None = None,
    collector_reflected: ArrayLike | None = None,
    particle_unfolded_end_positions: ArrayLike | None = None,
    collector_unfolded_end_positions: ArrayLike | None = None,
    arena_size: tuple[float, float] | None = None,
    time_tolerance: float = 1e-12,
    distance_tolerance: float = 1e-12,
) -> SweptCaptureResult:
    """Resolve earliest continuous contacts for primary fixed geometry.

    With unfolded proposed endpoints and ``arena_size``, rectangular specular
    paths are decomposed at every wall hit (including arbitrary multi-wall
    overshoot and simultaneous corner hits), and contact is solved on the union
    of particle and collector path segments. Without unfolded inputs, only an
    unreflected chord is accepted; declaring a reflection without its unfolded
    endpoint raises rather than silently falling back to endpoint capture.

    Ownership is selected by earliest contact fraction, then closest chord
    distance. Remaining exact ties use a key derived only from scenario seed,
    step index, and stable particle ID, so unrelated events and array order
    cannot shift the choice.
    """
    particle_start = np.asarray(particle_start_positions, dtype=np.float64)
    particle_end = np.asarray(particle_end_positions, dtype=np.float64)
    collector_start = np.asarray(collector_start_positions, dtype=np.float64)
    collector_end = np.asarray(collector_end_positions, dtype=np.float64)
    if (
        particle_start.ndim != 2
        or particle_start.shape[1] != 2
        or particle_end.shape != particle_start.shape
    ):
        raise ValueError("particle start/end positions must have shape (N, 2)")
    if (
        collector_start.ndim != 2
        or collector_start.shape[1] != 2
        or collector_end.shape != collector_start.shape
    ):
        raise ValueError("collector start/end positions must have shape (M, 2)")
    particle_count = particle_start.shape[0]
    collector_count = collector_start.shape[0]
    if state.owner.shape != (particle_count,):
        raise ValueError("owner array does not match particle count")
    if len(state.offsets) != collector_count:
        raise ValueError("offset list does not match collector count")
    contact_radius = collector_radius + particle_radius
    if not np.isfinite(contact_radius) or contact_radius < 0:
        raise ValueError("capture radii must be finite and non-negative")
    if event_seed < 0 or step_index < 0:
        raise ValueError("event_seed and step_index must be non-negative")

    stable_particles = (
        np.arange(particle_count, dtype=np.int64)
        if particle_ids is None
        else np.asarray(particle_ids, dtype=np.int64)
    )
    stable_collectors = (
        np.arange(collector_count, dtype=np.int64)
        if collector_ids is None
        else np.asarray(collector_ids, dtype=np.int64)
    )
    if stable_particles.shape != (particle_count,) or np.unique(stable_particles).size != particle_count:
        raise ValueError("particle_ids must contain N unique stable IDs")
    if (
        stable_collectors.shape != (collector_count,)
        or np.unique(stable_collectors).size != collector_count
        or not np.array_equal(np.sort(stable_collectors), np.arange(collector_count))
    ):
        raise ValueError("collector_ids must be a permutation of 0..M-1")
    reflected_particles = (
        np.zeros(particle_count, dtype=np.bool_)
        if particle_reflected is None
        else np.asarray(particle_reflected, dtype=np.bool_)
    )
    reflected_collectors = (
        np.zeros(collector_count, dtype=np.bool_)
        if collector_reflected is None
        else np.asarray(collector_reflected, dtype=np.bool_)
    )
    if reflected_particles.shape != (particle_count,):
        raise ValueError("particle_reflected must have shape (N,)")
    if reflected_collectors.shape != (collector_count,):
        raise ValueError("collector_reflected must have shape (M,)")

    exact_reflected_inputs = (
        particle_unfolded_end_positions is not None
        or collector_unfolded_end_positions is not None
        or arena_size is not None
    )
    if exact_reflected_inputs and (
        particle_unfolded_end_positions is None
        or collector_unfolded_end_positions is None
        or arena_size is None
    ):
        raise ValueError(
            "particle/collector unfolded endpoints and arena_size must be supplied together"
        )
    if exact_reflected_inputs:
        particle_unfolded_end = np.asarray(
            particle_unfolded_end_positions, dtype=np.float64
        )
        collector_unfolded_end = np.asarray(
            collector_unfolded_end_positions, dtype=np.float64
        )
        arena = np.asarray(arena_size, dtype=np.float64)
        if particle_unfolded_end.shape != particle_start.shape:
            raise ValueError("particle unfolded endpoints must have shape (N, 2)")
        if collector_unfolded_end.shape != collector_start.shape:
            raise ValueError("collector unfolded endpoints must have shape (M, 2)")
        if (
            arena.shape != (2,)
            or not np.all(np.isfinite(arena))
            or np.any(arena <= 0)
        ):
            raise ValueError("arena_size must contain two finite positive values")
        expected_particle_end = np.asarray(
            reflect(particle_unfolded_end, arena), dtype=np.float64
        )
        expected_collector_end = np.asarray(
            reflect(collector_unfolded_end, arena), dtype=np.float64
        )
        if not np.allclose(
            particle_end, expected_particle_end, atol=distance_tolerance, rtol=0.0
        ):
            raise ValueError(
                "particle endpoints do not match reflected unfolded proposals"
            )
        if not np.allclose(
            collector_end, expected_collector_end, atol=distance_tolerance, rtol=0.0
        ):
            raise ValueError(
                "collector endpoints do not match reflected unfolded proposals"
            )
    else:
        if np.any(reflected_particles) or np.any(reflected_collectors):
            raise ValueError(
                "reflected paths require unfolded endpoints and arena_size"
            )
        particle_unfolded_end = particle_end
        collector_unfolded_end = collector_end
        arena = None

    events: list[ContactEvent] = []
    decisions: list[TieDecision] = []
    guarded_pairs = 0
    row_by_stable_particle = {
        int(stable_id): row for row, stable_id in enumerate(stable_particles)
    }
    for stable_particle_id in np.sort(stable_particles):
        particle_row = row_by_stable_particle[int(stable_particle_id)]
        if state.owner[particle_row] >= 0:
            continue
        candidates: list[tuple[int, float, float]] = []
        for collector_row, stable_collector_id in enumerate(stable_collectors):
            if exact_reflected_inputs:
                assert arena is not None
                contact = _piecewise_reflected_first_contact(
                    particle_start[particle_row],
                    particle_unfolded_end[particle_row],
                    collector_start[collector_row],
                    collector_unfolded_end[collector_row],
                    arena,
                    contact_radius,
                    time_tolerance=time_tolerance,
                    distance_tolerance=distance_tolerance,
                )
            else:
                contact = _first_contact_fraction(
                    particle_start[particle_row],
                    particle_end[particle_row],
                    collector_start[collector_row],
                    collector_end[collector_row],
                    contact_radius,
                    tolerance=max(time_tolerance, distance_tolerance),
                )
            if contact is not None:
                candidates.append((int(stable_collector_id), *contact))
        if not candidates:
            continue

        earliest = min(item[1] for item in candidates)
        time_tied = [
            item for item in candidates if abs(item[1] - earliest) <= time_tolerance
        ]
        closest = min(item[2] for item in time_tied)
        finalists = np.array(
            [
                item[0]
                for item in time_tied
                if abs(item[2] - closest) <= distance_tolerance
            ],
            dtype=np.int64,
        )
        if finalists.size == 1:
            chosen = int(finalists[0])
        else:
            chosen = _event_keyed_choice(
                finalists,
                event_seed=event_seed,
                step_index=step_index,
                particle_id=int(stable_particle_id),
            )
            decisions.append(
                TieDecision(
                    step_index=step_index,
                    particle_id=int(stable_particle_id),
                    candidate_collector_ids=tuple(int(value) for value in np.sort(finalists)),
                    chosen_collector_id=chosen,
                )
            )
        state.owner[particle_row] = chosen
        events.append(
            ContactEvent(
                particle_id=int(stable_particle_id),
                collector_id=chosen,
                time_fraction=earliest,
                closest_distance=closest,
            )
        )

    return SweptCaptureResult(
        events=tuple(events),
        tie_decisions=tuple(decisions),
        guarded_reflection_pairs=guarded_pairs,
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
