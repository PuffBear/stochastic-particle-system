"""Bounded holonomic dynamics for collector agents."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .boundaries import reflect


def bounded_velocity(
    actions: ArrayLike, *, max_speed: float
) -> NDArray[np.float64]:
    """Map normalized two-dimensional actions to speed-bounded velocities.

    Actions inside the unit Euclidean ball are scaled by ``max_speed``;
    actions outside it are projected onto the ball.  This keeps direction
    while imposing the same physical speed limit on every collector.
    """
    action = np.asarray(actions, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 2:
        raise ValueError("actions must have shape (M, 2)")
    if not np.all(np.isfinite(action)):
        raise ValueError("actions must be finite")
    if not np.isfinite(max_speed) or max_speed < 0:
        raise ValueError("max_speed must be finite and non-negative")

    norm = np.linalg.norm(action, axis=1, keepdims=True)
    unit_ball = action / np.maximum(norm, 1.0)
    return max_speed * unit_ball


def advance_collectors(
    positions: ArrayLike,
    actions: ArrayLike,
    *,
    dt: float,
    max_speed: float,
    arena_size: tuple[float, float],
) -> NDArray[np.float64]:
    """Advance collectors for one reflected-boundary holonomic step."""
    pos = np.asarray(positions, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must have shape (M, 2)")
    if action.shape != pos.shape:
        raise ValueError("actions must match collector positions")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    arena = np.asarray(arena_size, dtype=np.float64)
    if arena.shape != (2,) or not np.all(np.isfinite(arena)) or np.any(arena <= 0):
        raise ValueError("arena_size must contain two finite positive values")

    proposed = pos + dt * bounded_velocity(action, max_speed=max_speed)
    return reflect(proposed, arena)
