"""Exact reflection in rectangular domains, including arbitrary overshoot."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def reflect(values: ArrayLike, length: float | ArrayLike) -> NDArray[np.float64]:
    """Reflect coordinates into ``[0, length]`` while preserving overshoot.

    The triangle-wave construction handles multiple wall crossings without a
    loop. ``length`` may be a scalar or broadcastable array such as ``[W, H]``.
    """
    x = np.asarray(values, dtype=np.float64)
    side = np.asarray(length, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(side)):
        raise ValueError("coordinates and side lengths must be finite")
    if np.any(side <= 0):
        raise ValueError("side lengths must be positive")
    period = 2.0 * side
    folded = np.mod(x, period)
    return np.where(folded <= side, folded, period - folded)

