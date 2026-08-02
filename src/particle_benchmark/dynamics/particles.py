"""Free-particle propagation with externally supplied stochastic forcing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .boundaries import reflect
from .fields import field_velocity


@dataclass(frozen=True)
class MatchedTrajectories:
    """Signal and null paths produced from identical initial state and noise."""

    null: NDArray[np.float64]
    signal: NDArray[np.float64]
    noise: NDArray[np.float64]


def advance_free_particles(
    positions: ArrayLike,
    epsilon: ArrayLike,
    *,
    dt: float,
    diffusion_sigma: float,
    arena_size: tuple[float, float],
    field_family: str,
    signal_strength: float,
    field_kwargs: dict[str, object] | None = None,
    return_unfolded: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance one time step under the benchmark's noise convention.

    Parameters
    ----------
    return_unfolded:
        If True, return (reflected_positions, unfolded_proposals) so callers
        can pass the unfolded positions to swept-capture geometry without
        duplicating the arithmetic and risking floating-point divergence.
    """
    pos = np.asarray(positions, dtype=np.float64)
    forcing = np.asarray(epsilon, dtype=np.float64)
    if pos.shape != forcing.shape or pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions and epsilon must both have shape (N, 2)")
    if dt <= 0 or diffusion_sigma < 0:
        raise ValueError("dt must be positive and diffusion_sigma non-negative")
    velocity = field_velocity(
        pos,
        field_family,
        signal_strength,
        **(field_kwargs or {}),
    )
    proposed = (
        pos
        + dt * velocity
        + diffusion_sigma * np.sqrt(dt) * forcing
    )
    reflected = reflect(proposed, np.asarray(arena_size, dtype=np.float64))
    if return_unfolded:
        return reflected, proposed
    return reflected


def simulate_matched_pair(
    initial_positions: ArrayLike,
    noise: ArrayLike,
    *,
    dt: float,
    diffusion_sigma: float,
    arena_size: tuple[float, float],
    field_family: str,
    signal_strength: float,
    field_kwargs: dict[str, object] | None = None,
) -> MatchedTrajectories:
    """Simulate a null/signal pair using the same supplied Brownian tensor."""
    initial = np.asarray(initial_positions, dtype=np.float64)
    forcing = np.asarray(noise, dtype=np.float64)
    if forcing.ndim != 3 or forcing.shape[1:] != initial.shape:
        raise ValueError("noise must have shape (H, N, 2)")

    null = np.empty((forcing.shape[0] + 1, *initial.shape), dtype=np.float64)
    signal = np.empty_like(null)
    null[0] = initial
    signal[0] = initial
    for step, epsilon in enumerate(forcing):
        null[step + 1] = advance_free_particles(
            null[step],
            epsilon,
            dt=dt,
            diffusion_sigma=diffusion_sigma,
            arena_size=arena_size,
            field_family=field_family,
            signal_strength=0.0,
            field_kwargs=field_kwargs,
        )
        signal[step + 1] = advance_free_particles(
            signal[step],
            epsilon,
            dt=dt,
            diffusion_sigma=diffusion_sigma,
            arena_size=arena_size,
            field_family=field_family,
            signal_strength=signal_strength,
            field_kwargs=field_kwargs,
        )
    return MatchedTrajectories(null=null, signal=signal, noise=forcing.copy())

