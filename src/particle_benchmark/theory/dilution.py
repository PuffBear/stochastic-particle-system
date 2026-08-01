"""SNR characterisation of the observation-overlap dilution trap.

Observation: in SPS-P02, four independent collectors (ρ=2) achieved a mean
yield gain of 0.01104, while a single collector achieved 0.06938 — a factor
of ~6 worse per agent despite covering more arena area.

The dilution trap arises because independent collectors may observe the same
particles redundantly rather than covering disjoint regions.  Each redundant
observation degrades the team's effective sensing budget without improving
field-direction estimation.

Proposition 3 (observation-overlap dilution).
Let M collectors be placed uniformly at random in an arena of area A, each
with sensing area a = π·r².  The expected fraction of the arena covered by
at least one collector is

    C(M) = 1 − (1 − a/A)^M   [inclusion-exclusion upper bound, exact under
                                independence and large A/a]

The expected fraction of unique coverage per total sensing budget M·a is

    η(M) = C(M) / (M · a/A)

Under η(M) < 1 (which holds whenever a/A > 0 and M > 1), independent agents
waste sensing budget on redundant observations.  A shared summary can signal
"I am already here" and redirect coverage, recovering a fraction (1 − η(M))
of wasted budget.

For the canonical task: a/A = π·0.16²/4 ≈ 0.0201, M=4:
    C(4) ≈ 1 − (1 − 0.0201)^4 ≈ 0.0778
    η(4) = 0.0778 / (4·0.0201) ≈ 0.967

The geometric overlap is small (~3.3% waste) and cannot explain a 6× dilution.
The primary dilution mechanism is therefore not spatial overlap but
*particle-sample fragmentation*: with M independent collectors each observing
K nearest particles, each agent's local sample is K out of ~N/M locally dense
particles, and agents with few field-aligned particles get noisy estimates
that cancel across the independent action decisions.

SNR scaling
-----------
Let ρ = α·√dt/σ be the dimensionless signal coordinate.  A single collector
observing K valid velocity observations accumulates a field-direction SNR of

    SNR₁ = ρ · √(K · f_valid)

per episode step.  M independent collectors with the same K each accumulate
the same SNR₁ per agent — but act on independent noisy estimates.  The team's
combined effective SNR for the *independent* action policy depends on the
correlation structure of their local estimates, not just the sum.

Under shared summary, each agent receives v̄ = (Σ_i v_i) / (M·K·f_valid),
which has SNR:

    SNR_shared = ρ · √(M · K · f_valid) = √M · SNR₁

So the shared summary achieves a √M SNR gain over any individual agent —
and over the independent policy when agents act on non-pooled estimates.
"""

from __future__ import annotations

import numpy as np


def overlap_probability(
    sensing_radius: float,
    arena_size: tuple[float, float],
    n_collectors: int,
) -> float:
    """Expected probability that a uniformly random point is covered by ≥2 collectors.

    Uses the inclusion-exclusion approximation valid when sensing area ≪ arena area.
    """
    area = arena_size[0] * arena_size[1]
    sensing_area = np.pi * sensing_radius**2
    coverage_fraction = sensing_area / area
    p_covered_by_at_least_one = 1.0 - (1.0 - coverage_fraction) ** n_collectors
    p_covered_by_at_least_two = (
        1.0 - (1.0 - coverage_fraction) ** n_collectors
        - n_collectors * coverage_fraction * (1.0 - coverage_fraction) ** (n_collectors - 1)
    )
    return float(max(0.0, p_covered_by_at_least_two))


def expected_unique_coverage(
    sensing_radius: float,
    arena_size: tuple[float, float],
    n_collectors: int,
) -> dict[str, object]:
    """Expected unique arena fraction covered and per-budget efficiency η(M).

    Parameters
    ----------
    sensing_radius:
        Radius r of each collector's sensing disc.
    arena_size:
        (width, height) of the rectangular arena.
    n_collectors:
        Team size M.

    Returns
    -------
    dict with coverage_fraction_per_collector, expected_unique_coverage_fraction,
    sensing_budget_efficiency, geometric_dilution_fraction.
    """
    area = arena_size[0] * arena_size[1]
    sensing_area = np.pi * sensing_radius**2
    cov_per = float(sensing_area / area)
    unique_cov = float(1.0 - (1.0 - cov_per) ** n_collectors)
    total_budget_fraction = n_collectors * cov_per
    efficiency = float(unique_cov / total_budget_fraction) if total_budget_fraction > 0 else 1.0

    return {
        "coverage_fraction_per_collector": cov_per,
        "expected_unique_coverage_fraction": unique_cov,
        "total_sensing_budget_fraction": total_budget_fraction,
        "sensing_budget_efficiency": efficiency,
        "geometric_dilution_fraction": float(1.0 - efficiency),
        "overlap_probability": overlap_probability(sensing_radius, arena_size, n_collectors),
    }


def snr_per_seed(
    rho: float,
    n_collectors: int,
    k_particles_per_collector: int,
    validity_fraction: float,
    shared: bool = True,
) -> dict[str, object]:
    """Field-direction estimation SNR per episode step.

    Parameters
    ----------
    rho:
        Dimensionless signal coordinate α·√dt/σ.
    n_collectors:
        Team size M.
    k_particles_per_collector:
        Sensing slot count K per collector.
    validity_fraction:
        Expected fraction of K slots with valid velocity observations.
    shared:
        If True, compute SNR under shared team-mean summary (all M·K·f_valid
        observations pooled).  If False, compute per-agent SNR (K·f_valid each).
    """
    n_valid_per_agent = k_particles_per_collector * validity_fraction
    n_valid_team = n_collectors * n_valid_per_agent

    n_eff = n_valid_team if shared else n_valid_per_agent
    snr = float(rho * np.sqrt(n_eff))

    return {
        "rho": rho,
        "shared": shared,
        "n_valid_per_agent": n_valid_per_agent,
        "n_valid_team_total": n_valid_team,
        "effective_n_for_snr": n_eff,
        "snr_per_step": snr,
    }


def dilution_snr_ratio(
    rho: float,
    n_collectors: int,
    k_particles_per_collector: int,
    validity_fraction: float,
) -> dict[str, object]:
    """SNR gain of shared summary over per-agent independent estimate.

    Returns the theoretical √M gain from pooling, and the canonical task values.
    """
    shared = snr_per_seed(rho, n_collectors, k_particles_per_collector, validity_fraction, shared=True)
    independent = snr_per_seed(rho, n_collectors, k_particles_per_collector, validity_fraction, shared=False)

    snr_ratio = (
        shared["snr_per_step"] / independent["snr_per_step"]
        if independent["snr_per_step"] > 0 else float("inf")
    )
    theoretical_ratio = float(np.sqrt(n_collectors))

    return {
        "n_collectors": n_collectors,
        "rho": rho,
        "shared_snr": shared["snr_per_step"],
        "independent_snr_per_agent": independent["snr_per_step"],
        "snr_ratio_shared_over_independent": snr_ratio,
        "theoretical_sqrt_M_ratio": theoretical_ratio,
        "note": (
            f"Sharing pooling all {n_collectors}·{k_particles_per_collector}·{validity_fraction:.2f} "
            f"observations achieves √{n_collectors}={theoretical_ratio:.2f}× higher "
            "field-direction SNR than any single independent agent."
        ),
    }
