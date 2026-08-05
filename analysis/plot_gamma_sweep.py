#!/usr/bin/env python3
"""Plot Γ-conservation sweep results.

Three-panel figure:
  Left:   Shared-arm reward curves y(L) for each ω (reward vs. L).
  Centre: L_max*(ω) vs. T_corr = 1/(ω·dt), with linear fit → slope = Γ.
  Right:  Γ(ω) = ω·dt·L_max*(ω) vs. ω — should be flat in linear regime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO    = Path(__file__).parent.parent
DATA    = REPO / "results" / "FR-B4" / "fr_b4_gamma_sweep.json"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

DT = 0.02

# Colour ramp: one hue per ω, light→dark blue-to-red via a diverging-ish qualitative order
COLORS = ["#1a6faf", "#3092c4", "#69b3d6", "#f4a261", "#e76f51", "#c1121f", "#780000"]


def fit_peak(Ls: np.ndarray, means: np.ndarray) -> float:
    """Fit a quadratic to the reward curve and return the peak L.

    Falls back to argmax if the quadratic has no interior peak.
    """
    Ls = np.asarray(Ls, dtype=float)
    means = np.asarray(means, dtype=float)
    # Fit quadratic in log(L) space (spreads points more evenly)
    logL = np.log(Ls)
    coeffs = np.polyfit(logL, means, 2)  # ax² + bx + c
    a, b, _ = coeffs
    if a < 0:   # concave → interior maximum at logL* = -b/(2a)
        logL_star = -b / (2 * a)
        L_star = float(np.exp(logL_star))
        # Clamp to observed range
        L_star = float(np.clip(L_star, Ls[0], Ls[-1]))
    else:
        # Convex — peak is at one of the endpoints (unusual; fall back to argmax)
        L_star = float(Ls[np.argmax(means)])
    return L_star


def main() -> None:
    data = json.loads(DATA.read_text())

    omega_grid = data["omega_grid"]
    l_grid     = data["l_grid"]
    cells      = data["cells"]

    # Restructure: rewards[omega][L] = mean
    rewards: dict[float, dict[int, float]] = {}
    ses:     dict[float, dict[int, float]] = {}
    for cell in cells:
        omega = cell["omega"]
        L     = cell["L"]
        if omega not in rewards:
            rewards[omega] = {}
            ses[omega] = {}
        rewards[omega][L] = cell["mean"]
        ses[omega][L]     = cell["se"]

    # Fit L_max* for each ω
    t_corr_vals: list[float] = []
    l_star_vals: list[float] = []
    gamma_vals:  list[float] = []

    for omega in omega_grid:
        t_corr = 1.0 / (omega * DT)
        Ls    = np.array(l_grid, dtype=float)
        means = np.array([rewards[omega][L] for L in l_grid])
        L_star = fit_peak(Ls, means)
        gamma  = omega * DT * L_star
        t_corr_vals.append(t_corr)
        l_star_vals.append(L_star)
        gamma_vals.append(gamma)
        print(f"  ω={omega:.2f}  T_corr={t_corr:.1f}  L_max*={L_star:.1f}  Γ={gamma:.3f}",
              file=sys.stderr)

    # Linear fit through origin: L_max* = Γ · T_corr
    t_corr_arr  = np.array(t_corr_vals)
    l_star_arr  = np.array(l_star_vals)
    gamma_fit   = float(np.sum(l_star_arr * t_corr_arr) / np.sum(t_corr_arr**2))
    gamma_mean  = float(np.mean(gamma_vals))
    gamma_sd    = float(np.std(gamma_vals, ddof=1))
    print(f"\nΓ (fit through origin) = {gamma_fit:.3f}", file=sys.stderr)
    print(f"Γ (pointwise mean ± sd) = {gamma_mean:.3f} ± {gamma_sd:.3f}", file=sys.stderr)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    ax_curves, ax_lmax, ax_gamma = axes

    # Panel A: reward curves
    Ls_arr = np.array(l_grid, dtype=float)
    for i, omega in enumerate(omega_grid):
        t_corr = 1.0 / (omega * DT)
        means  = np.array([rewards[omega][L] for L in l_grid])
        errs   = np.array([ses[omega][L]     for L in l_grid])
        color  = COLORS[i % len(COLORS)]
        ax_curves.plot(l_grid, means, "o-", color=color, lw=1.6, ms=4,
                       label=rf"$\omega={omega}$ ($T_{{\rm c}}={t_corr:.0f}$)")
        ax_curves.fill_between(l_grid, means - errs, means + errs,
                               color=color, alpha=0.12)
    ax_curves.set_xlabel("Window length $L$", fontsize=9)
    ax_curves.set_ylabel("Shared-arm captures", fontsize=9)
    ax_curves.set_title("Reward vs. $L$ by $\\omega$", fontsize=9)
    ax_curves.legend(fontsize=6.5, loc="lower left", ncol=1)
    ax_curves.tick_params(labelsize=8)
    ax_curves.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    # Panel B: L_max* vs. T_corr
    t_fine = np.linspace(0, max(t_corr_vals) * 1.05, 200)
    ax_lmax.plot(t_fine, gamma_fit * t_fine, "k--", lw=1.2,
                 label=rf"$L^* = {gamma_fit:.2f}\,T_{{\rm corr}}$")
    for i, (tc, ls, omega) in enumerate(zip(t_corr_vals, l_star_vals, omega_grid)):
        ax_lmax.scatter([tc], [ls], color=COLORS[i % len(COLORS)], zorder=3, s=40,
                        label=rf"$\omega={omega}$")
    ax_lmax.set_xlabel(r"$T_{\rm corr} = 1/(\omega\,{\rm d}t)$ (steps)", fontsize=9)
    ax_lmax.set_ylabel(r"$L^*_{\max}(\omega)$  (steps)", fontsize=9)
    ax_lmax.set_title(r"$L^*_{\max}$ vs.\ $T_{\rm corr}$", fontsize=9)
    ax_lmax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax_lmax.set_xlim(left=0)
    ax_lmax.set_ylim(bottom=0)
    ax_lmax.tick_params(labelsize=8)

    # Panel C: Γ(ω)
    ax_gamma.axhline(gamma_mean, color="gray", lw=1.0, ls="--",
                     label=rf"mean $\Gamma={gamma_mean:.2f}$")
    ax_gamma.fill_between([min(omega_grid)*0.85, max(omega_grid)*1.15],
                          gamma_mean - gamma_sd, gamma_mean + gamma_sd,
                          color="gray", alpha=0.10, label=r"$\pm1$ SD")
    for i, (omega, gamma) in enumerate(zip(omega_grid, gamma_vals)):
        ax_gamma.scatter([omega], [gamma], color=COLORS[i % len(COLORS)], zorder=3, s=50)
    ax_gamma.set_xlabel(r"$\omega$ (rad/step)", fontsize=9)
    ax_gamma.set_ylabel(r"$\Gamma(\omega) = \omega\,{\rm d}t\cdot L^*_{\max}$", fontsize=9)
    ax_gamma.set_title(r"Dimensionless group $\Gamma$ vs.\ $\omega$", fontsize=9)
    ax_gamma.set_xlim(min(omega_grid)*0.85, max(omega_grid)*1.15)
    ax_gamma.legend(fontsize=7.5, loc="upper right")
    ax_gamma.tick_params(labelsize=8)

    n_seeds = data["seeds"]
    s0, s1  = data["seeds"][0], data["seeds"][-1]
    fig.suptitle(
        rf"$\Gamma$-conservation: $\alpha={data['alpha']}$, $\sigma={data['diffusion_sigma']}$, "
        rf"$M={data['collector_count']}$; seeds {s0}–{s1} ($n={n_seeds}$). "
        rf"Fit: $\Gamma = {gamma_fit:.3f}$, pointwise $\bar{{\Gamma}} = {gamma_mean:.3f} \pm {gamma_sd:.3f}$.",
        fontsize=8,
    )

    pdf = FIG_DIR / "fig_gamma_sweep.pdf"
    png = FIG_DIR / "fig_gamma_sweep.png"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Saved {pdf}", file=sys.stderr)
    print(f"Saved {png}", file=sys.stderr)
    plt.close(fig)


if __name__ == "__main__":
    main()
