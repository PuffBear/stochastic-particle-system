#!/usr/bin/env python3
"""Generate fig_dispersion: angular bias & inter-agent dispersion vs. L.

Two-row figure comparing slow (omega=1.5) and mid (omega=5.0) conditions.
Reads the two JSON files produced by run_fr_b4_dispersion.py.
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
SLOW_DATA = REPO / "results" / "FR-B4" / "fr_b4_dispersion_slow_v2.json"
MID_DATA  = REPO / "results" / "FR-B4" / "fr_b4_dispersion_mid.json"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BLUE   = "#1f77b4"   # shared arm
ORANGE = "#d62728"   # independent arm
ALPHA_FILL = 0.15
LW = 1.8
MS = 5


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_row(ax_bias, ax_disp, data: dict, lmax: int) -> None:
    omega  = data["omega"]
    t_corr = data["t_corr"]
    sums   = data["summaries"]

    Ls      = np.array([s["L"] for s in sums], dtype=float)
    sb_mean = np.array([s["shared_abs_mean"] for s in sums])
    sb_se   = np.array([s["shared_abs_se"]   for s in sums])
    ib_mean = np.array([s["indep_abs_mean"]  for s in sums])
    ib_se   = np.array([s["indep_abs_se"]    for s in sums])
    id_mean = np.array([s["indep_inter_disp_mean"] for s in sums])
    id_se   = np.array([s["indep_inter_disp_se"]   for s in sums])

    dt = 0.02
    lag_pred = omega * (Ls - 1) * dt / 2.0

    # --- Bias panel ---
    ax_bias.plot(Ls, sb_mean, "o-", color=BLUE,   lw=LW, ms=MS, label="Shared arm")
    ax_bias.fill_between(Ls, sb_mean - sb_se, sb_mean + sb_se,
                         color=BLUE, alpha=ALPHA_FILL)
    ax_bias.plot(Ls, ib_mean, "s--", color=ORANGE, lw=LW, ms=MS, label="Independent arm")
    ax_bias.fill_between(Ls, ib_mean - ib_se, ib_mean + ib_se,
                         color=ORANGE, alpha=ALPHA_FILL)
    ax_bias.plot(Ls, lag_pred, "k:", lw=1.1, label=r"$\omega(L{-}1)dt/2$")
    ax_bias.axvline(lmax, color="gray", lw=0.8, ls="--")
    ylim = ax_bias.get_ylim()
    ax_bias.text(lmax + Ls[-1] * 0.02, ylim[1] * 0.93,
                 r"$L_{\max}$", fontsize=8, color="gray", va="top")
    ax_bias.set_ylabel("Mean angular error (rad)", fontsize=9)
    ax_bias.set_title(
        rf"$\omega={omega}$ ($T_{{\rm corr}}={t_corr:.0f}$): angular bias",
        fontsize=9
    )
    ax_bias.legend(fontsize=7.5, loc="upper left")
    ax_bias.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax_bias.tick_params(labelsize=8)

    # --- Dispersion panel ---
    ax_disp.plot(Ls, np.zeros_like(Ls), "o-", color=BLUE, lw=LW, ms=MS,
                 label="Shared arm (0 by construction)")
    ax_disp.plot(Ls, id_mean, "s--", color=ORANGE, lw=LW, ms=MS,
                 label="Independent arm")
    ax_disp.fill_between(Ls, np.maximum(id_mean - id_se, 0), id_mean + id_se,
                         color=ORANGE, alpha=ALPHA_FILL)
    ax_disp.axvline(lmax, color="gray", lw=0.8, ls="--")
    ylim = ax_disp.get_ylim()
    ax_disp.text(lmax + Ls[-1] * 0.02, max(id_mean) * 0.93,
                 r"$L_{\max}$", fontsize=8, color="gray", va="top")
    ax_disp.set_ylabel("Inter-agent circ. std (rad)", fontsize=9)
    ax_disp.set_title(
        rf"$\omega={omega}$ ($T_{{\rm corr}}={t_corr:.0f}$): inter-agent dispersion",
        fontsize=9
    )
    ax_disp.legend(fontsize=7.5, loc="upper right")
    ax_disp.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax_disp.tick_params(labelsize=8)
    ax_disp.set_ylim(bottom=0)

    for ax in (ax_bias, ax_disp):
        ax.set_xlabel("Window length $L$ (steps)", fontsize=9)


def main() -> None:
    slow = load(SLOW_DATA)
    mid  = load(MID_DATA)

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0), constrained_layout=True)

    plot_row(axes[0, 0], axes[0, 1], slow, lmax=10)
    plot_row(axes[1, 0], axes[1, 1], mid,  lmax=3)

    n_slow = len(slow["seeds"])
    n_mid  = len(mid["seeds"])
    s0, s1 = slow["seeds"][0], slow["seeds"][-1]
    m0, m1 = mid["seeds"][0],  mid["seeds"][-1]
    fig.suptitle(
        rf"Directional diagnostics: slow ($\omega=1.5$, seeds {s0}–{s1}, $n={n_slow}$) "
        rf"and mid ($\omega=5.0$, seeds {m0}–{m1}, $n={n_mid}$). Bands: $\pm1$ SE.",
        fontsize=8,
    )

    pdf_path = FIG_DIR / "fig_dispersion.pdf"
    png_path = FIG_DIR / "fig_dispersion.png"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved {pdf_path}", file=sys.stderr)
    print(f"Saved {png_path}", file=sys.stderr)
    plt.close(fig)


if __name__ == "__main__":
    main()
