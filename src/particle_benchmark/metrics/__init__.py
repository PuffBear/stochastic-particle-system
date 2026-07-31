"""Scientific metrics with explicit event-stage separation."""

from .episode import (
    first_interception_step,
    matched_first_interception_gain,
    per_step_signal_to_noise,
)

__all__ = [
    "first_interception_step",
    "matched_first_interception_gain",
    "per_step_signal_to_noise",
]
