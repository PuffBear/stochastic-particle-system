"""Core deterministic and stochastic dynamics primitives."""

from .boundaries import reflect
from .capture import CaptureState, resolve_captures
from .fields import field_velocity
from .particles import MatchedTrajectories, simulate_matched_pair

__all__ = [
    "CaptureState",
    "MatchedTrajectories",
    "field_velocity",
    "reflect",
    "resolve_captures",
    "simulate_matched_pair",
]
"""Physical dynamics used by the particle benchmark."""

from .collectors import advance_collectors, bounded_velocity

__all__ = ["advance_collectors", "bounded_velocity"]
