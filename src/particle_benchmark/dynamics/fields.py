"""Latent transport fields for the first benchmark release."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def field_velocity(
    positions: ArrayLike,
    family: str,
    signal_strength: float,
    *,
    orientation: float = 0.0,
    centre: tuple[float, float] = (0.5, 0.5),
    vortex_scale: float = 0.25,
    clockwise: bool = False,
    epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Return deterministic field velocity with shape ``(N, 2)``."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if not np.isfinite(signal_strength) or signal_strength < 0:
        raise ValueError("signal_strength must be finite and non-negative")

    if family == "null":
        return np.zeros_like(pos)
    if family == "uniform":
        direction = np.array([np.cos(orientation), np.sin(orientation)])
        return np.broadcast_to(signal_strength * direction, pos.shape).copy()
    if family != "vortex":
        raise ValueError(f"unknown field family: {family}")
    if vortex_scale <= 0 or not np.isfinite(vortex_scale):
        raise ValueError("vortex_scale must be finite and positive")

    centre_array = np.asarray(centre, dtype=np.float64)
    if centre_array.shape != (2,):
        raise ValueError("centre must be a pair")
    radius = pos - centre_array
    norm = np.linalg.norm(radius, axis=1, keepdims=True)
    tangent = np.concatenate((-radius[:, 1:2], radius[:, 0:1]), axis=1)
    tangent = tangent / (norm + epsilon)
    envelope = np.exp(-(norm**2) / (2.0 * vortex_scale**2))
    sign = -1.0 if clockwise else 1.0
    velocity = sign * signal_strength * envelope * tangent
    velocity[norm[:, 0] <= epsilon] = 0.0
    return velocity

