"""Coupled Brownian utilities for timestep-refinement experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def aggregate_brownian_increments(
    finest: ArrayLike, factor: int
) -> NDArray[np.float64]:
    """Sum finest *standard Brownian increments* into exact coarse increments."""
    values = np.asarray(finest, dtype=np.float64)
    if values.ndim < 1 or factor <= 0 or values.shape[0] % factor:
        raise ValueError("time axis must be divisible by a positive factor")
    return values.reshape(values.shape[0] // factor, factor, *values.shape[1:]).sum(axis=1)


def coupled_standard_normals(
    seed: int, *, finest_steps: int, shape_per_step: tuple[int, ...], factor: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return fine normals and coarse normals representing the same Brownian path."""
    if finest_steps <= 0:
        raise ValueError("finest_steps must be positive")
    fine = np.random.default_rng(seed).standard_normal((finest_steps, *shape_per_step))
    increments = fine / np.sqrt(float(finest_steps))
    coarse_increments = aggregate_brownian_increments(increments, factor)
    coarse = coarse_increments * np.sqrt(float(finest_steps / factor))
    return fine, coarse
