#!/usr/bin/env python3
"""Generate fig_dispersion: angular bias & inter-agent dispersion vs. L for slow condition.

Reads the JSON produced by run_fr_b4_dispersion.py and saves:
  figures/fig_dispersion.pdf  (for the paper)
  figures/fig_dispersion.png  (for the artifact page)
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
DATA = REPO / "results" / "FR-B4" / "fr_b4_dispersion_slow_v2.json"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BLUE   = "#1f77b4"
ORANGE = "#d62728"   # red-orange for independent arm (warm vs. cool)
ALPHA_FILL = 0.15
LW = 1.8
CAP = 3


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    data = load(DATA)
    summaries = data["summaries"]
    omega = data["omega"]
    t_corr = data["t_corr"]

    Ls       = np.array([s["L"] for s in summaries], dtype=float)
    # Angular bias (mean |error|) — shared arm
    sb_mean  = np.array([s["shared_abs_mean"] for s in summaries])
    sb_se    = np.array([s["shared_abs_se"]   for s in summaries])
    # Angular bias — independent arm
    ib_mean  = np.array([s["indep_abs_mean"]  for s in summaries])
    ib_se    = np.array([s["indep_abs_se"]    for s in summaries])
    # Inter-agent dispersion — shared arm (always 0 by construction)
    sd_mean  = np.zeros_like(Ls)        # shared_inter_disp = 0
    # Inter-agent dispersion — independent arm
    id_mean  = np.array([s["indep_inter_disp_mean"] for s in summaries])
    id_se    = np.array([s["indep_inter_disp_se"]   for s in summaries])

    # Predicted angular lag: omega * (L-1) * dt / 2, dt=0.02
    dt = 0.02
    lag_pred = omega * (Ls - 1) * dt / 2.0

    # -----------------------------------------------------------------------
    # Figure: two panels side by side
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2),
                                   constrained_layout=True)

    # --- Panel A: Angular bias ---
    ax1.plot(Ls, sb_mean, "o-", color=BLUE,   lw=LW, ms=5, label="Shared arm")
    ax1.fill_between(Ls,
                     sb_mean - sb_se,
                     sb_mean + sb_se,
                     color=BLUE, alpha=ALPHA_FILL)
    ax1.plot(Ls, ib_mean, "s--", color=ORANGE, lw=LW, ms=5, label="Independent arm")
    ax1.fill_between(Ls,
                     ib_mean - ib_se,
                     ib_mean + ib_se,
                     color=ORANGE, alpha=ALPHA_FILL)
    # Predicted lag
    ax1.plot(Ls, lag_pred, "k:", lw=1.1, label=r"$\omega(L{-}1)dt/2$ (predicted lag)")

    # L_max marker
    ax1.axvline(10, color="gray", lw=0.8, ls="--")
    ax1.text(10 + 0.6, ax1.get_ylim()[1] * 0.93 if ax1.get_ylim()[1] > 0.3 else 0.7,
             r"$L_{\max}$", fontsize=8, color="gray", va="top")

    ax1.set_xlabel("Window length $L$ (steps)", fontsize=9)
    ax1.set_ylabel("Mean angular error (rad)", fontsize=9)
    ax1.set_title(r"(a) Angular bias $|\hat\theta - \theta_{\rm true}|$", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax1.tick_params(labelsize=8)

    # --- Panel B: Inter-agent dispersion ---
    # Shared: exactly 0
    ax2.plot(Ls, sd_mean, "o-", color=BLUE, lw=LW, ms=5, label="Shared arm (0 by construction)")
    # Independent
    ax2.plot(Ls, id_mean, "s--", color=ORANGE, lw=LW, ms=5, label="Independent arm")
    ax2.fill_between(Ls,
                     np.maximum(id_mean - id_se, 0),
                     id_mean + id_se,
                     color=ORANGE, alpha=ALPHA_FILL)

    ax2.axvline(10, color="gray", lw=0.8, ls="--")
    ax2.text(10 + 0.6, max(id_mean) * 0.93 if len(id_mean) > 0 else 0.5,
             r"$L_{\max}$", fontsize=8, color="gray", va="top")

    ax2.set_xlabel("Window length $L$ (steps)", fontsize=9)
    ax2.set_ylabel("Inter-agent circular std (rad)", fontsize=9)
    ax2.set_title(r"(b) Inter-agent directional dispersion", fontsize=9)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax2.tick_params(labelsize=8)
    ax2.set_ylim(bottom=0)

    fig.suptitle(
        rf"$\omega={omega}$ ($T_{{\rm corr}}={t_corr:.0f}$ steps), {data['seeds'][0]}–{data['seeds'][-1]+1} seeds",
        fontsize=9,
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
