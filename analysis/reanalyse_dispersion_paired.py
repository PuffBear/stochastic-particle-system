#!/usr/bin/env python3
"""Paired reanalysis of angular-error comparisons for Figure 3.

Re-runs episodes for the five L levels needed for the three adjacent
comparisons, stores per-seed shared angular error, and reports:
  - paired t-test (one-sided, df=n-1=31)
  - paired-seed bootstrap 95% CI for the mean difference (B=10000)

Comparisons:
  Slow (omega=1.5, L_max=10):  E(5) > E(10),  E(20) > E(10)
  Mid  (omega=5.0, L_max=3):   E(10) > E(3)

Seeds match the main experiment: 9001–9032, n=32.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or analysis/
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis"))

from run_fr_b4_dispersion import run_episode_instrumented   # noqa: E402

SEEDS    = list(range(9001, 9033))   # 32 seeds — identical to main experiment
N_BOOT   = 10_000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Scipy-free t-distribution p-value (Numerical Recipes betainc)
# ---------------------------------------------------------------------------
def _betacf(a: float, b: float, x: float, max_iter: int = 300) -> float:
    fpmin = 1e-30
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin: d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin: c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin: d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin: c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-7:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front_ab = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front_ab * _betacf(a, b, x)
    front_ba = math.exp(b * math.log(1.0 - x) + a * math.log(x) - lbeta) / b
    return 1.0 - front_ba * _betacf(b, a, 1.0 - x)


def t_p_onesided(t: float, df: int) -> float:
    """P(T_{df} >= t) for t > 0 (one-sided upper tail)."""
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x) / 2.0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def per_seed_mean_abs_error(seeds: list[int], L: int, omega: float) -> np.ndarray:
    """Per-seed mean |angular error| for the shared arm."""
    out = []
    for seed in seeds:
        ep = run_episode_instrumented(seed, L, omega=omega)
        true   = np.array(ep["theta_true"])
        shared = np.array(ep["theta_shared"])
        T = len(true)
        errs = np.array([(shared[t] - true[t] + np.pi) % (2 * np.pi) - np.pi
                         for t in range(T)])
        out.append(float(np.nanmean(np.abs(errs))))
    return np.array(out)


def paired_analysis(label: str, a: np.ndarray, b: np.ndarray,
                    La: int, Lb: int, rng: np.random.Generator) -> None:
    """Paired t-test + bootstrap for H1: mean(a) > mean(b)."""
    d = a - b
    n = len(d)
    mean_d = float(np.mean(d))
    sd_d   = float(np.std(d, ddof=1))
    se_d   = sd_d / np.sqrt(n)
    t_stat = mean_d / se_d
    df     = n - 1
    p_val  = t_p_onesided(t_stat, df)

    boot = np.array([np.mean(rng.choice(d, size=n, replace=True))
                     for _ in range(N_BOOT)])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    r = float(np.corrcoef(a, b)[0, 1])

    print(f"\n  [{label}]  H1: E(L={La}) > E(L={Lb})")
    print(f"    n={n}  mean E({La})={np.mean(a):.4f}  mean E({Lb})={np.mean(b):.4f}")
    print(f"    mean diff = {mean_d:+.4f}  SD(diff) = {sd_d:.4f}  r = {r:.3f}")
    print(f"    paired t({df}) = {t_stat:.3f}   one-sided p = {p_val:.4f}")
    print(f"    bootstrap 95% CI for mean diff: [{ci_lo:+.4f}, {ci_hi:+.4f}]")


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    print(f"Paired dispersion reanalysis  n={len(SEEDS)}  B={N_BOOT}\n",
          file=sys.stderr)

    # ── Slow (omega=1.5, L_max=10) ──────────────────────────────────────────
    print("Running slow (omega=1.5) ...", file=sys.stderr)
    slow = {}
    for L in [5, 10, 20]:
        print(f"  L={L}", file=sys.stderr, end="  ", flush=True)
        slow[L] = per_seed_mean_abs_error(SEEDS, L, 1.5)
        print(f"mean={slow[L].mean():.4f}", file=sys.stderr)

    print("\n=== SLOW ===")
    paired_analysis("Slow", slow[5],  slow[10],  5, 10, rng)
    paired_analysis("Slow", slow[20], slow[10], 20, 10, rng)

    # ── Mid (omega=5.0, L_max=3) ─────────────────────────────────────────────
    print("\nRunning mid (omega=5.0) ...", file=sys.stderr)
    mid = {}
    for L in [3, 10]:
        print(f"  L={L}", file=sys.stderr, end="  ", flush=True)
        mid[L] = per_seed_mean_abs_error(SEEDS, L, 5.0)
        print(f"mean={mid[L].mean():.4f}", file=sys.stderr)

    print("\n=== MID ===")
    paired_analysis("Mid", mid[10], mid[3], 10, 3, rng)

    print("\nDone.")


if __name__ == "__main__":
    main()
