"""Small policies used to validate environment plumbing, not as results."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dynamics.fields import field_velocity


def stationary_policy(collector_count: int) -> NDArray[np.float64]:
    """Return zero action for every collector."""
    if collector_count <= 0:
        raise ValueError("collector_count must be positive")
    return np.zeros((collector_count, 2), dtype=np.float64)


def random_policy(
    collector_count: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Sample actions uniformly by direction and within unit-disc area."""
    if collector_count <= 0:
        raise ValueError("collector_count must be positive")
    angle = rng.uniform(0.0, 2.0 * np.pi, size=collector_count)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=collector_count))
    return radius[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))


def random_action_tensor(
    seed: int, *, horizon: int, collector_count: int
) -> NDArray[np.float64]:
    """Pre-generate a complete random-policy action tensor for matched pairs."""
    if horizon <= 0 or collector_count <= 0:
        raise ValueError("horizon and collector_count must be positive")
    from .seeding import make_streams

    rng = make_streams(seed).policy
    angle = rng.uniform(0.0, 2.0 * np.pi, size=(horizon, collector_count))
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=(horizon, collector_count)))
    return radius[..., None] * np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def privileged_field_policy(
    collector_positions: ArrayLike,
    *,
    field_family: str,
    signal_strength: float,
    field_kwargs: dict[str, object] | None = None,
) -> NDArray[np.float64]:
    """Move upstream against the true field, intentionally using privilege."""
    positions = np.asarray(collector_positions, dtype=np.float64)
    velocity = field_velocity(
        positions,
        field_family,
        signal_strength,
        **(field_kwargs or {}),
    )
    norm = np.linalg.norm(velocity, axis=1, keepdims=True)
    return -np.divide(velocity, norm, out=np.zeros_like(velocity), where=norm > 0)
