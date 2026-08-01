"""Pure, capacity-matched SPS-C04 communication interventions.

Each row is one sender's complete per-step message
``(velocity_x, velocity_y, valid_fraction)``.  No observation, policy, random
stream, or agent identity enters these functions: the independent and
all-to-all arms differ only in the aggregation map.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


AggregationMode = Literal["independent", "all_to_all"]


def validate_three_scalar_messages(messages: ArrayLike) -> NDArray[np.float64]:
    """Validate and copy an exact-three-scalar message matrix.

    Columns 0--1 are bounded velocity components in ``[-1, 1]`` and column 2
    is a valid-observation fraction in ``[0, 1]``.  Returning a copy prevents
    either intervention from mutating the sender messages.
    """
    array = np.asarray(messages, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] != 3:
        raise ValueError("messages must have shape (sender_count, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("messages must contain only finite values")
    if np.any(np.abs(array[:, :2]) > 1.0):
        raise ValueError("velocity message components must lie in [-1, 1]")
    if np.any(array[:, 2] < 0.0) or np.any(array[:, 2] > 1.0):
        raise ValueError("valid fractions must lie in [0, 1]")
    return array.copy()


def aggregate_three_scalar_messages(
    messages: ArrayLike,
    *,
    mode: AggregationMode,
) -> NDArray[np.float64]:
    """Return one capacity-matched three-scalar input per receiving agent.

    ``independent`` is the identity map: receiver ``i`` uses sender ``i``'s
    local summary. ``all_to_all`` broadcasts the arithmetic mean of every
    sender's complete three-scalar summary to every receiver.  Both outputs
    have shape ``(M, 3)`` and therefore expose exactly three controller inputs
    per receiver.
    """
    validated = validate_three_scalar_messages(messages)
    if mode == "independent":
        return validated
    if mode == "all_to_all":
        # Sorting each scalar channel first makes floating-point accumulation
        # independent of sender enumeration while preserving the arithmetic
        # mean exactly as a mathematical intervention.
        team_mean = np.mean(
            np.sort(validated, axis=0), axis=0, dtype=np.float64
        )
        return np.broadcast_to(team_mean, validated.shape).copy()
    raise ValueError(f"unknown aggregation mode: {mode}")
