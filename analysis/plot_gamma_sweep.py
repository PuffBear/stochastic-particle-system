#!/usr/bin/env python3
"""Plot Γ-conservation sweep results.

Three-panel figure:
  Left:   Shared-arm reward curves y(L) for each ω (reward vs. L).
  Centre: L_opt(ω) vs. T_corr = 1/(ω·dt), with linear fit → slope = Γ.
  Right:  Γ(ω) = ω·dt·L_opt(ω) vs. ω — with bootstrap 95% CIs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO    = Path(__file__).parent.parent
DATA    = REPO / "results" / "FR-B4" / "fr_b4_gamma_sweep.json"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

DT          = 0.02
N_BOOTSTRAP = 4000
RNG_SEED    = 42

# Regime classification based on the main experiment findings.
# linear: L_opt falls cleanly in (1, H) with negligible sinc attenuation.
# anomalous: long-episode, sinc-attenuation, or rapid-rotation regime.
REGIME = {
    0.50: "linear",
    0.75: "linear-bdy",  # boundary-censored (L_opt may exceed grid)
    1.00: "long-ep",     # anomalous: only L=1 significant in main exp
    1.50: "linear",      # confirmed interior anchor
    2.50: "sinc-att",    # anomalous: sinc-attenuation
    5.00: "linear",      # confirmed interior anchor
    8.00: "rapid-rot",   # anomalous: rapid-rotation
}

# Colours: linear-regime in blues, anomalous in grays/reds
COLOR = {
    "linear":     "#1f77b4",
    "linear-bdy": "#69b3d6",
    "long-ep":    "#d62728",
    "sinc-att":   "#ff7f0e",
    "rapid-rot":  "#9467bd",
}
MARKER = {
    "linear": "o", "linear-bdy": "s",
    "long-ep": "^", "sinc-att": "D", "rapid-rot": "v",
}


def argmax_L(Ls: np.ndarray, means: np.ndarray) -> float:
    """Return L at the maximum reward."""
    idx = int(np.argmax(means))
    return float(Ls[idx])


def bootstrap_gamma_ci(
    captures_by_L: dict[int, list[int]],
    l_grid: list[int],
    omega: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for Γ = ω·dt·L_opt via seed resampling.

    Returns (gamma_point, ci_lo, ci_hi).
    """
    Ls = np.array(l_grid, dtype=float)
    # Stack captures into (n_seeds, n_L) array
    n_seeds = len(next(iter(captures_by_L.values())))
    cap_mat = np.array([captures_by_L[L] for L in l_grid], dtype=float).T  # (n_seeds, n_L)

    point_means = cap_mat.mean(axis=0)
    gamma_point = omega * DT * argmax_L(Ls, point_means)

    boot_gammas = np.empty(n_boot)
    idx_all = np.arange(n_seeds)
    for b in range(n_boot):
        idx = rng.choice(idx_all, size=n_seeds, replace=True)
        boot_means = cap_mat[idx].mean(axis=0)
        boot_gammas[b] = omega * DT * argmax_L(Ls, boot_means)

    ci_lo = float(np.percentile(boot_gammas, 2.5))
    ci_hi = float(np.percentile(boot_gammas, 97.5))
    return gamma_point, ci_lo, ci_hi


def main() -> None:
    data = json.loads(DATA.read_text())

    omega_grid = data["omega_grid"]
    l_grid     = data["l_grid"]
    cells      = data["cells"]

    # Restructure: rewards[omega][L] = mean, se; captures_raw[omega][L] = list
    rewards:      dict[float, dict[int, float]]      = {}
    ses:          dict[float, dict[int, float]]      = {}
    captures_raw: dict[float, dict[int, list[int]]]  = {}
    for cell in cells:
        omega = cell["omega"]
        L     = cell["L"]
        if omega not in rewards:
            rewards[omega]      = {}
            ses[omega]          = {}
            captures_raw[omega] = {}
        rewards[omega][L]      = cell["mean"]
        ses[omega][L]          = cell["se"]
        captures_raw[omega][L] = cell["captures"]

    rng = np.random.default_rng(RNG_SEED)

    # Identify L_opt per ω and bootstrap Γ CIs
    results = []
    for omega in omega_grid:
        t_corr = 1.0 / (omega * DT)
        Ls     = np.array(l_grid, dtype=float)
        means  = np.array([rewards[omega][L] for L in l_grid])
        L_star = argmax_L(Ls, means)
        gamma, ci_lo, ci_hi = bootstrap_gamma_ci(
            captures_raw[omega], l_grid, omega, N_BOOTSTRAP, rng
        )
        regime = REGIME.get(omega, "unknown")
        results.append(dict(
            omega=omega, t_corr=t_corr, L_star=L_star,
            gamma=gamma, ci_lo=ci_lo, ci_hi=ci_hi, regime=regime,
        ))
        print(f"  ω={omega:.2f}  T_corr={t_corr:.1f}  L_opt={L_star:.1f}"
              f"  Γ={gamma:.3f}  95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]  [{regime}]",
              file=sys.stderr)

    # Fit only on confirmed linear-regime points (not boundary-censored)
    linear = [r for r in results if r["regime"] == "linear"]
    tc_lin = np.array([r["t_corr"] for r in linear])
    ls_lin = np.array([r["L_star"] for r in linear])
    gamma_fit = float(np.dot(ls_lin, tc_lin) / np.dot(tc_lin, tc_lin))
    gamma_lin_mean = float(np.mean([r["gamma"] for r in linear]))
    print(f"\nΓ (linear-regime fit through origin) = {gamma_fit:.3f}", file=sys.stderr)
    print(f"Γ (linear-regime pointwise mean)     = {gamma_lin_mean:.3f}", file=sys.stderr)
    for r in linear:
        print(f"  ω={r['omega']:.2f}  Γ={r['gamma']:.3f}  "
              f"95% CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", file=sys.stderr)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    ax_curves, ax_lmax, ax_gamma = axes

    # Panel A: reward curves, coloured by regime
    for r in results:
        omega  = r["omega"]
        t_corr = r["t_corr"]
        regime = r["regime"]
        means  = np.array([rewards[omega][L] for L in l_grid])
        errs   = np.array([ses[omega][L]     for L in l_grid])
        color  = COLOR[regime]
        ls_style = "-" if "linear" in regime else "--"
        ax_curves.plot(l_grid, means, f"o{ls_style}", color=color, lw=1.5, ms=4,
                       label=rf"$\omega={omega}$ ($T_{{\rm c}}={t_corr:.0f}$) [{regime}]")
        ax_curves.fill_between(l_grid, means - errs, means + errs,
                               color=color, alpha=0.10)
    ax_curves.set_xlabel("Window length $L$", fontsize=9)
    ax_curves.set_ylabel("Shared-arm captures", fontsize=9)
    ax_curves.set_title(r"Shared-arm reward vs.\ $L$", fontsize=9)
    ax_curves.legend(fontsize=5.8, loc="lower left", ncol=1)
    ax_curves.tick_params(labelsize=8)
    ax_curves.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))

    # Panel B: L_opt vs. T_corr — fit line on linear regime only
    t_fine = np.linspace(0, max(r["t_corr"] for r in results) * 1.05, 200)
    ax_lmax.plot(t_fine, gamma_fit * t_fine, "k--", lw=1.2, zorder=0,
                 label=rf"$L_{{\rm opt}} = {gamma_fit:.2f}\,T_{{\rm corr}}$ (linear only)")
    for r in results:
        ax_lmax.scatter([r["t_corr"]], [r["L_star"]],
                        color=COLOR[r["regime"]], marker=MARKER[r["regime"]],
                        zorder=3, s=55,
                        label=rf"$\omega={r['omega']}$ [{r['regime']}]")
    ax_lmax.set_xlabel(r"$T_{\rm corr}$ (steps)", fontsize=9)
    ax_lmax.set_ylabel(r"$L_{\rm opt}(\omega)$  (steps)", fontsize=9)
    ax_lmax.set_title(r"$L_{\rm opt}$ vs.\ $T_{\rm corr}$", fontsize=9)
    ax_lmax.legend(fontsize=5.8, loc="upper left", ncol=1)
    ax_lmax.set_xlim(left=0)
    ax_lmax.set_ylim(bottom=0)
    ax_lmax.tick_params(labelsize=8)

    # Panel C: omega*dt*L_opt vs. omega, with bootstrap 95% CI error bars
    ax_gamma.axhline(gamma_fit, color="#1f77b4", lw=1.2, ls="--",
                     label=rf"linear-regime fit $= {gamma_fit:.2f}$")
    ax_gamma.axhspan(gamma_fit * 0.8, gamma_fit * 1.2, color="#1f77b4",
                     alpha=0.08, label=r"$\pm20\%$ band")
    for r in results:
        yerr_lo = r["gamma"] - r["ci_lo"]
        yerr_hi = r["ci_hi"] - r["gamma"]
        ax_gamma.errorbar(
            [r["omega"]], [r["gamma"]],
            yerr=[[yerr_lo], [yerr_hi]],
            fmt=MARKER[r["regime"]],
            color=COLOR[r["regime"]],
            ecolor=COLOR[r["regime"]], elinewidth=1.0, capsize=3,
            zorder=3, ms=6,
        )
    # Add 95% CI legend entry (one dummy line)
    ax_gamma.errorbar([], [], yerr=[[]], fmt="none",
                      ecolor="gray", elinewidth=1.0, capsize=3,
                      label="bootstrap 95% CI ($n{=}16$, $B{=}4000$)")
    ax_gamma.set_xlabel(r"$\omega$ (rad/step)", fontsize=9)
    ax_gamma.set_ylabel(r"$\Gamma = \omega\,dt \cdot L_{\rm opt}$", fontsize=9)
    ax_gamma.set_title(r"$\Gamma(\omega) = \omega\,dt \cdot L_{\rm opt}(\omega)$", fontsize=9)
    omegas = [r["omega"] for r in results]
    ax_gamma.set_xlim(min(omegas) * 0.7, max(omegas) * 1.15)
    ax_gamma.set_ylim(bottom=0)
    ax_gamma.legend(fontsize=7.0, loc="upper left")
    ax_gamma.tick_params(labelsize=8)

    n_seeds = len(data["seeds"])
    s0, s1  = data["seeds"][0], data["seeds"][-1]
    fig.suptitle(
        rf"Reward-optimal window sweep ($L_{{\rm opt}}$, not $L_{{\rm max}}$): "
        rf"$\alpha={data['alpha']}$, $\sigma={data['diffusion_sigma']}$, "
        rf"$M={data['collector_count']}$; seeds {s0}–{s1} ($n={n_seeds}$).  "
        rf"Linear-regime fit: $\Gamma={gamma_fit:.2f}$.  "
        r"Error bars: bootstrap 95\% CI.  Non-linear points excluded from fit.",
        fontsize=7.5,
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
