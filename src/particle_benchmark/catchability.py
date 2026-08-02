"""Dimensionless parameterization for the FR-B3 catchability study.

For a square or fixed-aspect-ratio arena with characteristic length ``L``,
the normalized one-step dynamics are determined by three transport groups:

``rho = alpha * sqrt(dt) / sigma``
    Drift-to-diffusion signal ratio per observation.
``kappa = alpha / v_max``
    Drift speed relative to collector speed (inverse control authority).
``eta = sigma * sqrt(dt) / L``
    Absolute diffusive displacement per step relative to the arena.

With horizon and normalized geometry fixed, ``(rho, kappa, eta)`` determine
the normalized drift, diffusion, and collector displacements.  ``(rho,
kappa)`` alone do not: changing ``eta`` changes all absolute step sizes while
leaving both ratios unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .environment import ParticleEnvConfig


@dataclass(frozen=True)
class CatchabilityGroups:
    """Dimensionless groups and normalized one-step displacement scales."""

    rho: float
    kappa: float
    eta: float
    normalized_drift_step: float
    normalized_control_step: float
    sensing_radius_ratio: float
    capture_radius_ratio: float
    aspect_ratio: float
    horizon_steps: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def characteristic_length(arena_size: tuple[float, float]) -> float:
    """Return the geometric-mean arena length used for nondimensionalization."""

    arena = np.asarray(arena_size, dtype=np.float64)
    if arena.shape != (2,) or not np.all(np.isfinite(arena)) or np.any(arena <= 0):
        raise ValueError("arena_size must contain two finite positive values")
    return float(math.sqrt(float(arena[0] * arena[1])))


def catchability_groups(config: ParticleEnvConfig) -> CatchabilityGroups:
    """Compute the FR-B3 groups from one validated environment config."""

    alpha = float(config.signal_strength)
    sigma = float(config.diffusion_sigma)
    speed = float(config.collector_max_speed)
    if alpha <= 0 or sigma <= 0 or speed <= 0:
        raise ValueError("FR-B3 requires positive signal, diffusion, and collector speed")
    length = characteristic_length(config.arena_size)
    sqrt_dt = math.sqrt(config.dt)
    rho = alpha * sqrt_dt / sigma
    kappa = alpha / speed
    eta = sigma * sqrt_dt / length
    return CatchabilityGroups(
        rho=rho,
        kappa=kappa,
        eta=eta,
        normalized_drift_step=alpha * config.dt / length,
        normalized_control_step=speed * config.dt / length,
        sensing_radius_ratio=config.sensing_radius / length,
        capture_radius_ratio=(config.collector_radius + config.particle_radius) / length,
        aspect_ratio=config.arena_size[0] / config.arena_size[1],
        horizon_steps=config.horizon,
    )


def physical_parameters_from_groups(
    *,
    rho: float,
    kappa: float,
    eta: float,
    dt: float,
    arena_size: tuple[float, float],
) -> dict[str, float]:
    """Recover ``alpha``, ``sigma``, and ``v_max`` from the three groups."""

    values = np.asarray([rho, kappa, eta, dt], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("rho, kappa, eta, and dt must be finite and positive")
    length = characteristic_length(arena_size)
    sigma = eta * length / math.sqrt(dt)
    alpha = rho * sigma / math.sqrt(dt)
    collector_max_speed = alpha / kappa
    return {
        "signal_strength": float(alpha),
        "diffusion_sigma": float(sigma),
        "collector_max_speed": float(collector_max_speed),
    }


def rescale_equivalent_config(
    config: ParticleEnvConfig, *, length_scale: float, time_scale: float
) -> ParticleEnvConfig:
    """Return a physically rescaled config with identical dimensionless groups.

    The number of simulation steps is unchanged.  Lengths scale by ``l`` and
    the duration of each step scales by ``t``.  Speeds therefore scale by
    ``l/t`` and the SDE diffusion coefficient by ``l/sqrt(t)``.
    """

    scales = np.asarray([length_scale, time_scale], dtype=np.float64)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("length_scale and time_scale must be finite and positive")
    speed_scale = length_scale / time_scale
    diffusion_scale = length_scale / math.sqrt(time_scale)
    return replace(
        config,
        arena_size=tuple(float(length_scale * x) for x in config.arena_size),
        dt=float(time_scale * config.dt),
        diffusion_sigma=float(diffusion_scale * config.diffusion_sigma),
        collector_max_speed=float(speed_scale * config.collector_max_speed),
        sensing_radius=float(length_scale * config.sensing_radius),
        collector_radius=float(length_scale * config.collector_radius),
        particle_radius=float(length_scale * config.particle_radius),
        signal_strength=float(speed_scale * config.signal_strength),
    )
