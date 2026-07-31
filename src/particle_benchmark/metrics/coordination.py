"""Coordination Efficiency (CE) metrics for MARL policies.

CE Framework
------------
CE_s(π) = [Y_s(π_comm) − Y_s(π_ablated)] / [Y_s(oracle) − Y_s(π_ablated)]

Where:
- Y_s(π_comm)    = unique team yield under the full policy with communication
- Y_s(π_ablated) = same architecture but communication ablated (messages zeroed)
- Y_s(oracle)    = full-state oracle yield (upper bound on per-seed performance)

Interpretation
--------------
CE > 0: communication helped relative to the ablated baseline.
CE = 0: communication was neutral (no improvement over the ablated policy).
CE = 1: communication fully closes the gap to oracle performance.
CE < 0: communication hurt performance (active interference / noise injection).

When oracle == ablated (no room to improve above ablated), CE is defined as 0
to avoid division by zero.

Field-Informative Seed Split
----------------------------
A seed s is *field-informative* if:
    oracle_yield(s) - stationary_yield(s) > threshold

This partitions seeds into two groups:
- Field-informative:   seeds where the advection field confers a meaningful
  advantage that a smart policy can exploit; these are the seeds where
  communication may allow coordinated upstream positioning.
- Field-uninformative: seeds where the random-walk component dominates the
  field; any CE advantage here is likely due to coincidence or memorisation.

Reporting CE separately on each partition clarifies whether an algorithm
exploits field structure (positive CE on informative seeds, near-zero on
uninformative) or injects noise (positive uninformative, zero informative).
"""
from __future__ import annotations

import numpy as np


def per_seed_ce(
    comm_yields: np.ndarray,
    ablated_yields: np.ndarray,
    oracle_yields: np.ndarray,
) -> np.ndarray:
    """Per-seed CE_s. Clips denominator to avoid division by zero.

    Parameters
    ----------
    comm_yields    : shape (n_seeds,) — yields under the full communication policy
    ablated_yields : shape (n_seeds,) — yields with communication ablated
    oracle_yields  : shape (n_seeds,) — oracle upper-bound yields

    Returns
    -------
    ce : shape (n_seeds,) float64 — CE_s for each seed.
         When oracle == ablated the denominator is effectively zero and CE_s = 0.
    """
    comm = np.asarray(comm_yields, dtype=np.float64)
    ablated = np.asarray(ablated_yields, dtype=np.float64)
    oracle = np.asarray(oracle_yields, dtype=np.float64)

    denom = oracle - ablated
    safe_denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)  # avoid division by zero
    raw_ce = (comm - ablated) / safe_denom
    ce = np.where(np.abs(denom) < 1e-12, 0.0, raw_ce)
    return ce.astype(np.float64)


def split_field_informative(
    oracle_yields: np.ndarray,
    stationary_yields: np.ndarray,
    threshold: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (informative_mask, uninformative_mask) boolean arrays.

    Field-informative seeds are those where the oracle substantially
    outperforms a stationary policy: oracle - stationary > threshold.
    Yields should be on a consistent scale (e.g. raw capture counts or
    fractions); the threshold must match that scale.

    Parameters
    ----------
    oracle_yields     : shape (n_seeds,) — oracle yield per seed
    stationary_yields : shape (n_seeds,) — stationary policy yield per seed
    threshold         : scalar — minimum oracle advantage for a seed to be
                        considered field-informative (default 4.0)

    Returns
    -------
    informative_mask   : boolean ndarray, True where oracle - stationary > threshold
    uninformative_mask : boolean ndarray, complement of informative_mask
    """
    oracle = np.asarray(oracle_yields, dtype=np.float64)
    stationary = np.asarray(stationary_yields, dtype=np.float64)
    informative = (oracle - stationary) > threshold
    return informative, ~informative


def ce_report(
    ce_values: np.ndarray,
    informative_mask: np.ndarray,
    algorithm_name: str,
) -> dict:
    """Returns a structured CE summary report.

    Parameters
    ----------
    ce_values        : shape (n_seeds,) from per_seed_ce
    informative_mask : shape (n_seeds,) boolean from split_field_informative
    algorithm_name   : str label for the algorithm

    Returns
    -------
    dict with keys:
        algorithm_name           : str
        all_seeds                : {mean, median, std, q10, q90, fraction_positive, n}
        field_informative_seeds  : {mean, median, std, fraction_positive, n}
        field_uninformative_seeds: {mean, median, std, fraction_positive, n}
        interpretation           : one of "exploits_field", "noise_injection",
                                   "harmful", "neutral"
    """
    ce = np.asarray(ce_values, dtype=np.float64)
    info_mask = np.asarray(informative_mask, dtype=bool)
    uninf_mask = ~info_mask

    def _stats(arr: np.ndarray) -> dict:
        n = len(arr)
        if n == 0:
            return {"mean": float("nan"), "median": float("nan"),
                    "std": float("nan"), "fraction_positive": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "fraction_positive": float(np.mean(arr > 0)),
            "n": n,
        }

    all_stats = _stats(ce)
    all_stats["q10"] = float(np.quantile(ce, 0.10)) if len(ce) > 0 else float("nan")
    all_stats["q90"] = float(np.quantile(ce, 0.90)) if len(ce) > 0 else float("nan")
    # Remove extra keys (q10, q90 are only in all_seeds)

    all_seeds_report = {
        "mean": all_stats["mean"],
        "median": all_stats["median"],
        "std": all_stats["std"],
        "q10": all_stats["q10"],
        "q90": all_stats["q90"],
        "fraction_positive": all_stats["fraction_positive"],
        "n": all_stats["n"],
    }

    ce_info = ce[info_mask]
    ce_uninf = ce[uninf_mask]
    info_report = _stats(ce_info)
    uninf_report = _stats(ce_uninf)

    # Determine interpretation.
    info_median = info_report["median"]
    uninf_mean = uninf_report["mean"]
    all_mean = all_seeds_report["mean"]

    if not np.isnan(info_median) and info_median > 0.1 and (
        np.isnan(uninf_mean) or uninf_mean < 0.05
    ):
        interpretation = "exploits_field"
    elif not np.isnan(uninf_mean) and uninf_mean > 0.05:
        interpretation = "noise_injection"
    elif not np.isnan(all_mean) and all_mean < 0:
        interpretation = "harmful"
    else:
        interpretation = "neutral"

    return {
        "algorithm_name": algorithm_name,
        "all_seeds": all_seeds_report,
        "field_informative_seeds": info_report,
        "field_uninformative_seeds": uninf_report,
        "interpretation": interpretation,
    }


def upstream_alignment_score(
    movement_directions: np.ndarray,
    field_direction: np.ndarray,
) -> dict:
    """Informational coordination measure based on upstream movement alignment.

    Computes cosine similarity between each collector's movement direction
    and the upstream direction (-field_direction, i.e. moving against drift
    to intercept incoming particles).

    Steps where all agents are stationary (zero movement) are excluded from
    the per-step aggregation; individual zero-movement agents are excluded
    per-agent.

    Parameters
    ----------
    movement_directions : shape (n_steps, n_agents, 2) — unit vectors of
                          step displacement for each agent at each step
    field_direction     : shape (2,) — unit vector of the field b_hat
                          (drift direction)

    Returns
    -------
    dict with keys:
        mean_alignment      : float — mean cosine similarity over all non-stationary
                              (step, agent) pairs
        per_agent_alignment : ndarray shape (n_agents,) — mean per agent
        per_step_alignment  : ndarray shape (n_steps,) — mean over agents per step
                              (NaN for steps where all agents are stationary)
        alignment_std       : float — std of cosine similarities
    """
    dirs = np.asarray(movement_directions, dtype=np.float64)  # (T, M, 2)
    field = np.asarray(field_direction, dtype=np.float64)  # (2,)

    # Upstream direction: against the drift.
    field_norm = np.linalg.norm(field)
    upstream = -field / (field_norm + 1e-9)  # (2,)

    n_steps, n_agents, _ = dirs.shape

    # Cosine similarity of each (step, agent) movement with upstream direction.
    # movement_directions are unit vectors (or zero for stationary agents).
    cos_sim = dirs @ upstream  # (T, M)

    # Mask out stationary agents (zero movement).
    move_norms = np.linalg.norm(dirs, axis=-1)  # (T, M)
    stationary = move_norms < 1e-9

    # Per-agent alignment: mean over steps where agent moves.
    per_agent = np.full(n_agents, float("nan"))
    for a in range(n_agents):
        active = ~stationary[:, a]
        if active.any():
            per_agent[a] = float(np.mean(cos_sim[active, a]))

    # Per-step alignment: mean over agents that move.
    per_step = np.full(n_steps, float("nan"))
    for t in range(n_steps):
        active = ~stationary[t, :]
        if active.any():
            per_step[t] = float(np.mean(cos_sim[t, active]))

    # Global mean and std: over all non-stationary (step, agent) pairs.
    active_all = ~stationary
    if active_all.any():
        active_cos = cos_sim[active_all]
        mean_align = float(np.mean(active_cos))
        std_align = float(np.std(active_cos))
    else:
        mean_align = float("nan")
        std_align = float("nan")

    return {
        "mean_alignment": mean_align,
        "per_agent_alignment": per_agent,
        "per_step_alignment": per_step,
        "alignment_std": std_align,
    }


def spatial_coverage_metrics(
    collector_positions: np.ndarray,
    sensing_radius: float = 0.16,
) -> dict:
    """Spatial coordination metrics computed from trajectory.

    Positions are assumed to be in [0, 1]^2 (unit arena). The arena is
    discretized into an 8×8 grid of cells to compute coverage entropy.

    Parameters
    ----------
    collector_positions : shape (n_steps, n_agents, 2)
    sensing_radius      : float — sensing radius for redundancy computation

    Returns
    -------
    dict with keys:
        mean_pairwise_distance : float — mean distance between all collector
                                 pairs, averaged over all steps
        redundancy_rate        : float — fraction of (step, pair) combos where
                                 distance < sensing_radius (overlapping sensing)
        coverage_entropy       : float — entropy of collector position
                                 distribution over the 8×8 arena grid
                                 (higher = more spatially spread out)
    """
    pos = np.asarray(collector_positions, dtype=np.float64)  # (T, M, 2)
    T, M, _ = pos.shape

    # --- Mean pairwise distance and redundancy rate ---
    pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
    if len(pairs) == 0:
        mean_pw_dist = 0.0
        redundancy_rate = 0.0
    else:
        n_pairs = len(pairs)
        pairwise_dists = np.zeros((T, n_pairs))
        for k, (i, j) in enumerate(pairs):
            pairwise_dists[:, k] = np.linalg.norm(pos[:, i, :] - pos[:, j, :], axis=-1)
        mean_pw_dist = float(np.mean(pairwise_dists))
        redundancy_rate = float(np.mean(pairwise_dists < sensing_radius))

    # --- Coverage entropy via 8×8 grid ---
    # Clip positions to [0, 1] to handle numerical edge cases.
    pos_clipped = np.clip(pos, 0.0, 1.0 - 1e-9)
    grid_size = 8
    cell_x = np.floor(pos_clipped[:, :, 0] * grid_size).astype(int)
    cell_y = np.floor(pos_clipped[:, :, 1] * grid_size).astype(int)
    # Flatten cell indices.
    cell_idx = cell_x * grid_size + cell_y  # (T, M), values in [0, 63]
    cell_flat = cell_idx.reshape(-1)  # (T*M,)

    counts = np.bincount(cell_flat, minlength=grid_size * grid_size).astype(np.float64)
    total = counts.sum()
    if total > 0:
        probs = counts[counts > 0] / total
        entropy = float(-np.sum(probs * np.log(probs)))
    else:
        entropy = 0.0

    return {
        "mean_pairwise_distance": mean_pw_dist,
        "redundancy_rate": redundancy_rate,
        "coverage_entropy": entropy,
    }
