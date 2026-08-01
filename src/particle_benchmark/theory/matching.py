"""Variance reduction from matched counterfactual pairing.

Proposition 1 (matched variance reduction).
Let Y_s(s) and Y_n(s) be the capture yields for seed s under signal and null
respectively. The matched estimator D̄ = (1/n)·Σ_s [Y_s(s) − Y_n(s)] has variance

    Var(D̄) = (1/n) · [Var(Y_s) + Var(Y_n) − 2·Cov(Y_s, Y_n)]
            = (1/n) · [Var(Y_s) + Var(Y_n)] · (1 − ρ_YY)

where ρ_YY = Corr(Y_s(s), Y_n(s)) is the within-seed correlation.  The unmatched
estimator (independent seeds for signal and null) has variance

    Var(Ȳ_s − Ȳ_n) = (1/n) · [Var(Y_s) + Var(Y_n)]

so the matched design reduces variance by a factor of (1 − ρ_YY), provided
ρ_YY > 0.  Under the matched counterfactual contract (shared initial state and
Brownian noise tensor), ρ_YY is expected to be positive because the hard and
easy seeds are the same in both conditions.

Corollary: the matched design is equivalent to an unmatched design with
n_eff = n / (1 − ρ_YY) seeds — a multiplicative gain in statistical efficiency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def matched_variance_reduction(
    signal_yields: ArrayLike,
    null_yields: ArrayLike,
) -> dict[str, object]:
    """Quantify variance reduction from matched counterfactual pairing.

    Parameters
    ----------
    signal_yields:
        Per-seed capture yields under signal (α > 0), shape (n,).
    null_yields:
        Per-seed capture yields under null (α = 0), *same seeds in same order*.

    Returns
    -------
    dict with:
        n_seeds                   number of paired seeds
        mean_difference           E[Y_signal − Y_null]  (the primary effect)
        matched_variance          Var(Y_signal − Y_null)   (matched design)
        unmatched_variance        Var(Y_signal) + Var(Y_null)  (independence bound)
        variance_reduction_factor (1 − ρ_YY); fraction of variance eliminated
        within_seed_correlation   ρ_YY = Corr(Y_signal, Y_null)
        matched_se                SE of the matched mean difference
        unmatched_se              SE that an unmatched design would achieve
        effective_n_equivalent    n_unmatched that equals matched_se
    """
    ys = np.asarray(signal_yields, dtype=float)
    yn = np.asarray(null_yields, dtype=float)
    if ys.shape != yn.shape or ys.ndim != 1:
        raise ValueError("signal_yields and null_yields must be 1-D arrays of equal length")
    n = len(ys)
    if n < 2:
        raise ValueError("need at least 2 paired seeds")

    diff = ys - yn
    matched_var = float(np.var(diff, ddof=1))
    var_s = float(np.var(ys, ddof=1))
    var_n = float(np.var(yn, ddof=1))
    unmatched_var = var_s + var_n

    corr = float(np.corrcoef(ys, yn)[0, 1]) if unmatched_var > 0 else 0.0
    reduction = float(1.0 - matched_var / unmatched_var) if unmatched_var > 0 else 0.0

    matched_se = float(np.sqrt(matched_var / n))
    unmatched_se = float(np.sqrt(unmatched_var / n))
    eff_n = float(unmatched_var / matched_var * n) if matched_var > 0 else float("inf")

    return {
        "n_seeds": n,
        "mean_difference": float(np.mean(diff)),
        "matched_variance": matched_var,
        "unmatched_variance": unmatched_var,
        "variance_reduction_factor": reduction,
        "within_seed_correlation": corr,
        "matched_se": matched_se,
        "unmatched_se": unmatched_se,
        "effective_n_equivalent": eff_n,
    }


def power_from_matching(
    n_seeds: int,
    mean_effect: float,
    matched_se_per_seed: float,
    alpha_one_sided: float = 0.05,
) -> float:
    """One-sided normal-approximation power for the matched t-test.

    Power = Φ(|μ_D| / (σ_D/√n) − z_α)

    Uses only the standard library ``math.erf`` — no scipy dependency.

    Parameters
    ----------
    n_seeds:
        Number of matched seed pairs.
    mean_effect:
        Expected value of D = Y_signal − Y_null (in the same units as yields).
    matched_se_per_seed:
        √Var(Y_signal − Y_null)  — the per-seed standard deviation of D,
        estimated from pilot data via ``matched_variance_reduction``.
    alpha_one_sided:
        One-sided type-I error rate.
    """
    import math

    if matched_se_per_seed <= 0:
        raise ValueError("matched_se_per_seed must be positive")
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _norm_ppf(p: float) -> float:
        # Rational approximation for Φ⁻¹(p), Abramowitz & Stegun 26.2.17
        # Accurate to |error| < 4.5e-4 over (0, 1).
        assert 0.0 < p < 1.0
        if p < 0.5:
            t = math.sqrt(-2.0 * math.log(p))
        else:
            t = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        num = c0 + c1 * t + c2 * t * t
        den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
        approx = t - num / den
        return approx if p >= 0.5 else -approx

    se = matched_se_per_seed / math.sqrt(n_seeds)
    z_alpha = _norm_ppf(1.0 - alpha_one_sided)
    ncp = abs(mean_effect) / se
    return float(_norm_cdf(ncp - z_alpha))
