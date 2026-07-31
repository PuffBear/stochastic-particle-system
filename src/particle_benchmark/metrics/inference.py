"""Seed-blocked inference for the preregistered detectability grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SimultaneousLowerBounds:
    """Observed grid means and a common studentized bootstrap correction."""

    mean: NDArray[np.float64]
    lower: NDArray[np.float64]
    critical_value: float
    bootstrap_draws: int
    confidence: float


def simultaneous_paired_lower_bounds(
    seed_by_grid_effects: ArrayLike,
    *,
    confidence: float = 0.95,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 73_031,
) -> SimultaneousLowerBounds:
    """Compute one-sided simultaneous studentized-bootstrap lower bounds.

    Rows are independent scenario-seed blocks and columns are nonzero signal
    grid points. Every bootstrap draw resamples complete rows, preserving the
    cross-grid matching. The common critical value is the requested quantile
    of the maximum, over grid points, of ``observed_mean-bootstrap_mean``.
    Subtracting it from every observed mean controls selection across the
    finite preregistered grid under the bootstrap approximation. Studentizing
    materially improves small-sample calibration relative to an unscaled
    percentile correction for the 12-seed pilot.
    """
    effects = np.asarray(seed_by_grid_effects, dtype=np.float64)
    if effects.ndim != 2 or effects.shape[0] < 2 or effects.shape[1] < 1:
        raise ValueError("effects must have shape (at least 2 seeds, grid points)")
    if not np.all(np.isfinite(effects)):
        raise ValueError("effects must be finite")
    if not 0.0 < confidence < 1.0 or bootstrap_draws < 100:
        raise ValueError("invalid confidence or insufficient bootstrap draws")

    # Canonicalize scenario-row order so a fixed bootstrap seed produces the
    # same result after an irrelevant input permutation.
    order = np.lexsort(tuple(effects[:, column] for column in reversed(range(effects.shape[1]))))
    effects = effects[order]
    observed = np.mean(effects, axis=0)
    observed_se = np.std(effects, axis=0, ddof=1) / np.sqrt(effects.shape[0])
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, effects.shape[0], size=(bootstrap_draws, effects.shape[0])
    )
    bootstrap_means = np.mean(effects[indices], axis=1)
    bootstrap_se = np.std(effects[indices], axis=1, ddof=1) / np.sqrt(
        effects.shape[0]
    )
    studentized_shortfall = np.divide(
        observed[None, :] - bootstrap_means,
        bootstrap_se,
        out=np.zeros_like(bootstrap_means),
        where=bootstrap_se > 1e-15,
    )
    max_shortfall = np.max(studentized_shortfall, axis=1)
    critical = float(np.quantile(max_shortfall, confidence, method="higher"))
    return SimultaneousLowerBounds(
        mean=observed,
        lower=observed - critical * observed_se,
        critical_value=critical,
        bootstrap_draws=bootstrap_draws,
        confidence=confidence,
    )


def grid_censored_boundary(
    rho_grid: ArrayLike, lower_bounds: ArrayLike
) -> dict[str, float | str | None]:
    """Apply the frozen first-positive-LCB rule without interpolation."""
    rho = np.asarray(rho_grid, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    if rho.ndim != 1 or lower.shape != rho.shape or rho.size == 0:
        raise ValueError("rho_grid and lower_bounds must be equal nonempty vectors")
    if not np.all(np.isfinite(rho)) or not np.all(np.diff(rho) > 0):
        raise ValueError("rho_grid must be finite and strictly increasing")
    crossing = np.flatnonzero(lower > 0.0)
    if crossing.size == 0:
        return {
            "status": "right_censored",
            "lower_rho": float(rho[-1]),
            "upper_rho": None,
        }
    index = int(crossing[0])
    if index == 0:
        return {
            "status": "left_censored",
            "lower_rho": None,
            "upper_rho": float(rho[0]),
        }
    return {
        "status": "interval",
        "lower_rho": float(rho[index - 1]),
        "upper_rho": float(rho[index]),
    }


def persistent_grid_censored_boundary(
    rho_grid: ArrayLike, lower_bounds: ArrayLike
) -> dict[str, float | str | None]:
    """Return the first grid point after which every LCB stays positive."""
    rho = np.asarray(rho_grid, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    if rho.ndim != 1 or lower.shape != rho.shape or rho.size == 0:
        raise ValueError("rho_grid and lower_bounds must be equal nonempty vectors")
    persistent = np.array([np.all(lower[index:] > 0.0) for index in range(rho.size)])
    if not np.any(persistent):
        return {"status": "right_censored", "lower_rho": float(rho[-1]), "upper_rho": None}
    index = int(np.flatnonzero(persistent)[0])
    if index == 0:
        return {"status": "left_censored", "lower_rho": None, "upper_rho": float(rho[0])}
    return {"status": "interval", "lower_rho": float(rho[index - 1]), "upper_rho": float(rho[index])}
