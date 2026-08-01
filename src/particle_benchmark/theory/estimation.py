"""Fisher information and team-mean sufficiency for field direction estimation.

Observation model
-----------------
Each valid apparent velocity observation takes the form

    v_i = α·b̂ + ε_i,    ε_i ~ N(0, σ²_obs·I₂)

where α is the field strength, b̂ ∈ S¹ is the (unknown) field direction, and
σ_obs = σ/√dt is the per-observation noise standard deviation that arises from
one-step finite differences of Brownian positions.

Proposition 2 (team-mean sufficiency).
Let v₁, …, v_N be N iid apparent velocity observations under the above model.
The statistic T = (Σ v_x, Σ v_y, N) — equivalently (v̄_x, v̄_y, N) — is
sufficient for the parameter (α·b̂_x, α·b̂_y) in the sense of the Fisher–
Neyman factorisation theorem.

Proof sketch: the joint density factors as
  p({v_i} | α·b̂) = (2πσ²_obs)^{-N} · exp[−Σ‖v_i − α·b̂‖²/(2σ²_obs)]
                  = h({v_i}) · g(Σv, N, α·b̂)
so (Σv_x, Σv_y, N) is sufficient.  The 3-number bounded summary (v̄_x, v̄_y,
f_valid) used in SPS-C03 is an equivalent sufficient statistic (scaling N by
the known slot count K gives f_valid = N/(M·K)).

Corollary: under Gaussian noise, no additional compression of M·K individual
velocity observations can extract more information about b̂ than the 3-number
team-mean summary.  The summary is an information-lossless compression.

Fisher information
------------------
The per-component Fisher information (treating b̂_x and b̂_y independently
under isotropy) from N valid observations is:

    I(α·b̂_j) = N / σ²_obs    (j ∈ {x, y})

The Cramér-Rao lower bound on MSE of the MLE v̄_j is σ²_obs / N.
"""

from __future__ import annotations

import numpy as np


def apparent_velocity_noise_std(diffusion_sigma: float, dt: float) -> float:
    """Noise std of a one-step apparent velocity under pure Brownian diffusion.

    One-step position increment: Δx ~ N(0, σ²·dt).
    Apparent velocity v = Δx/dt ~ N(0, σ²/dt).
    Noise std = σ/√dt.
    """
    if diffusion_sigma <= 0 or dt <= 0:
        raise ValueError("diffusion_sigma and dt must be positive")
    return float(diffusion_sigma / np.sqrt(dt))


def fisher_information_field_direction(
    signal_strength: float,
    obs_noise_std: float,
    n_valid_observations: float,
) -> dict[str, object]:
    """Fisher information for one component of α·b̂ from n_valid observations.

    Parameters
    ----------
    signal_strength:
        α — field strength.
    obs_noise_std:
        σ_obs = σ/√dt — per-observation noise standard deviation.
    n_valid_observations:
        Number of valid apparent velocity observations (can be fractional
        when averaging over many seeds / configurations).

    Returns
    -------
    dict with fisher_information_per_component and cramer_rao_bound_mse.
    """
    if obs_noise_std <= 0:
        raise ValueError("obs_noise_std must be positive")
    if n_valid_observations < 0:
        raise ValueError("n_valid_observations must be non-negative")

    snr = float(signal_strength / obs_noise_std)
    fisher = float(n_valid_observations / obs_noise_std**2)
    crb_mse = float(1.0 / fisher) if fisher > 0 else float("inf")
    crb_rmse = float(np.sqrt(crb_mse)) if fisher > 0 else float("inf")

    return {
        "signal_to_noise_ratio": snr,
        "n_valid_observations": n_valid_observations,
        "obs_noise_std": obs_noise_std,
        "fisher_information_per_component": fisher,
        "cramer_rao_bound_mse": crb_mse,
        "cramer_rao_bound_rmse": crb_rmse,
    }


def team_sensing_summary(
    signal_strength: float,
    diffusion_sigma: float,
    dt: float,
    n_collectors: int,
    k_particles_per_collector: int,
    validity_fraction: float,
) -> dict[str, object]:
    """Compare Fisher information: single agent vs. full team sharing.

    For a team of M collectors each observing up to K nearest particles with
    validity fraction f_valid:

        n_valid_team  = M · K · f_valid      (shared estimate)
        n_valid_agent = K · f_valid           (individual estimate)

    The team-mean summary (v̄_x, v̄_y, f_valid) pools all M·K·f_valid
    observations into 3 scalars without any information loss (Proposition 2).

    Parameters
    ----------
    validity_fraction:
        Expected fraction of K-particle slots with valid velocity estimates.
        Depends on particle density within sensing radius and history length.
    """
    if not (0.0 <= validity_fraction <= 1.0):
        raise ValueError("validity_fraction must be in [0, 1]")
    if n_collectors < 1 or k_particles_per_collector < 1:
        raise ValueError("collector count and K must be at least 1")

    obs_noise = apparent_velocity_noise_std(diffusion_sigma, dt)
    n_valid_agent = k_particles_per_collector * validity_fraction
    n_valid_team = n_collectors * n_valid_agent

    fi_agent = fisher_information_field_direction(signal_strength, obs_noise, n_valid_agent)
    fi_team = fisher_information_field_direction(signal_strength, obs_noise, n_valid_team)

    sharing_gain = (
        fi_team["fisher_information_per_component"]
        / fi_agent["fisher_information_per_component"]
        if fi_agent["fisher_information_per_component"] > 0
        else float("inf")
    )

    return {
        "obs_noise_std": obs_noise,
        "n_valid_per_agent": n_valid_agent,
        "n_valid_team_total": n_valid_team,
        "single_agent_fisher": fi_agent["fisher_information_per_component"],
        "single_agent_crb_rmse": fi_agent["cramer_rao_bound_rmse"],
        "team_shared_fisher": fi_team["fisher_information_per_component"],
        "team_shared_crb_rmse": fi_team["cramer_rao_bound_rmse"],
        "sharing_fisher_gain": sharing_gain,
        "sharing_rmse_gain": (
            fi_agent["cramer_rao_bound_rmse"] / fi_team["cramer_rao_bound_rmse"]
            if fi_team["cramer_rao_bound_rmse"] > 0 else float("inf")
        ),
        "sufficiency_holds": True,
        "sufficiency_note": (
            "Under isotropic Gaussian noise, (v̄_x, v̄_y, f_valid) is "
            "sufficient for (α·b̂_x, α·b̂_y): no information is lost "
            "by compressing M·K individual velocities to 3 scalars."
        ),
    }
