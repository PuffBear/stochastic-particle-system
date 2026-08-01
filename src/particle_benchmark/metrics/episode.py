"""Pre-contact and first-contact metrics used by the primary question."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def per_step_signal_to_noise(
    signal_strength: float, dt: float, diffusion_sigma: float
) -> float:
    """Dimensionless uniform-drift displacement relative to Brownian scale."""
    if signal_strength < 0 or dt <= 0 or diffusion_sigma <= 0:
        raise ValueError(
            "signal_strength must be non-negative; dt and sigma must be positive"
        )
    return float(signal_strength * np.sqrt(dt) / diffusion_sigma)


def first_interception_step(new_captures_by_step: Sequence[int]) -> int | None:
    """Return the one-based first step with a capture, or ``None``."""
    for zero_based_step, count in enumerate(new_captures_by_step):
        if count < 0:
            raise ValueError("capture counts cannot be negative")
        if count > 0:
            return zero_based_step + 1
    return None


def matched_first_interception_gain(
    signal_steps: Sequence[int | None],
    null_steps: Sequence[int | None],
    horizon: int,
) -> np.ndarray:
    """Paired improvement in normalized time-to-first-interception.

    Steps are one-based in ``[1, horizon]``. A missing interception is encoded
    as ``horizon + 1``. Positive values mean the signal episode intercepted
    earlier than its matched null episode. Dividing by ``horizon`` makes the
    result the fraction of the episode horizon saved.
    """
    if horizon <= 0 or len(signal_steps) != len(null_steps):
        raise ValueError("invalid horizon or unmatched pair count")
    sentinel = horizon + 1
    signal = np.array(
        [sentinel if value is None else value for value in signal_steps],
        dtype=np.float64,
    )
    null = np.array(
        [sentinel if value is None else value for value in null_steps],
        dtype=np.float64,
    )
    if np.any(signal < 0) or np.any(null < 0):
        raise ValueError("first-interception steps cannot be negative")
    if np.any(signal > sentinel) or np.any(null > sentinel):
        raise ValueError("first-interception steps exceed the horizon")
    return (null - signal) / horizon
