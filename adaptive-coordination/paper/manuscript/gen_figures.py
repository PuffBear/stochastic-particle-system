"""Generate figures for FR-B4 AAMAS manuscript.

Outputs (in same directory as this script):
  fig_crossover.pdf   — Δ̄ vs L for slow and mid (main Figure 1)
  fig_scaling.pdf     — L_max vs T_corr scaling fit (main Figure 2)
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Output directory ─────────────────────────────────────────────────────────
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ── Style constants (AAMAS two-column: max 3.33 in wide) ─────────────────────
# Categorical palette — two hues, CVD-safe (blue + orange)
# Validated against WCAG contrast on white (#FFFFFF):
#   #1A6FBF (blue)  L*≈37, contrast ~5.8:1 ✓
#   #C45E00 (orange) L*≈44, contrast ~4.6:1 ✓
# ΔE(blue, orange) in OKLab ≈ 45 >> 8 threshold ✓
COL_SLOW = "#1A6FBF"   # blue  — slow (T_corr=33)
COL_MID  = "#C45E00"   # orange — mid  (T_corr=10)

MARKER_SIG  = "o"   # filled circle — significant (p < 0.05)
MARKER_NS   = "o"   # same shape, open — not significant
MARK_SZ     = 5
LINE_W      = 1.5
ERR_CAP     = 2.5
ERR_LW      = 0.9
FONT_MAIN   = 8      # pt — body labels
FONT_TICK   = 7
FONT_LEGEND = 7

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         FONT_MAIN,
    "axes.titlesize":    FONT_MAIN,
    "axes.labelsize":    FONT_MAIN,
    "xtick.labelsize":   FONT_TICK,
    "ytick.labelsize":   FONT_TICK,
    "legend.fontsize":   FONT_LEGEND,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.linewidth":    0.4,
    "grid.color":        "#CCCCCC",
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype":      42,   # embed fonts as Type 42 (TrueType) for ACM
    "ps.fonttype":       42,
})

# ── Data ─────────────────────────────────────────────────────────────────────
L_VALS = [1, 3, 5, 10, 20, 30, 45, 67]

# slow (ω=1.5, T_corr=33) — window method, n=32
slow_dbar = [+1.469, +1.500, +1.656, +0.938, -0.375, -0.281, -0.469, -0.438]
slow_se   = [ 0.514,  0.414,  0.453,  0.566,  0.468,  0.522,  0.591,  0.625]
slow_sig  = [  True,   True,   True,   True,  False,  False,  False,  False]

# mid (ω=5.0, T_corr=10) — window method, n=32
mid_dbar  = [+0.875, +0.906, +0.219, +0.250, +0.562, +0.156, -0.250, +0.031]
mid_se    = [ 0.386,  0.337,  0.408,  0.535,  0.464,  0.502,  0.414,  0.429]
mid_sig   = [  True,   True,  False,  False,  False,  False,  False,  False]

L_MAX_SLOW = 10
L_MAX_MID  =  3

CI_MULT = 1.96   # 95% CI half-width

# ── Helper: draw one series ───────────────────────────────────────────────────
def _plot_series(ax, L_vals, dbar, se, sig, col, label, lmax):
    L = np.array(L_vals)
    d = np.array(dbar)
    e = np.array(se)
    s = np.array(sig, dtype=bool)

    # Line (full series, thin)
    ax.plot(L, d, color=col, lw=LINE_W, zorder=2)

    # Error bars
    ax.errorbar(L, d, yerr=CI_MULT * e,
                fmt="none", color=col, lw=ERR_LW, capsize=ERR_CAP,
                capthick=ERR_LW, zorder=3)

    # Significant markers (filled)
    ax.scatter(L[s], d[s], color=col, marker=MARKER_SIG, s=MARK_SZ**2,
               zorder=5, linewidths=0, label=f"{label} (sig.)")

    # Non-significant markers (open, white fill, colored edge)
    ax.scatter(L[~s], d[~s], facecolors="white", edgecolors=col,
               marker=MARKER_NS, s=MARK_SZ**2, lw=0.8,
               zorder=5, label=f"{label} (n.s.)")

    # L_max tick — small vertical bracket at top of plot
    ax.axvline(lmax, color=col, lw=0.7, ls="--", alpha=0.55, zorder=1)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: crossover plot
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.33, 2.55))

# Zero reference
ax.axhline(0, color="#888888", lw=0.7, ls="-", zorder=1)

_plot_series(ax, L_VALS, slow_dbar, slow_se, slow_sig,
             COL_SLOW, r"slow ($T_{\rm corr}{=}33$)", L_MAX_SLOW)
_plot_series(ax, L_VALS, mid_dbar,  mid_se,  mid_sig,
             COL_MID,  r"mid ($T_{\rm corr}{=}10$)",  L_MAX_MID)

# L_max labels
ax.text(L_MAX_SLOW + 1.5, -1.55,
        r"$L_{\max}{=}10$", color=COL_SLOW, fontsize=6.5,
        ha="left", va="top")
ax.text(L_MAX_MID  + 0.4, -1.55,
        r"$L_{\max}{=}3$", color=COL_MID, fontsize=6.5,
        ha="left", va="top")

# Axes
ax.set_xlabel(r"Memory window $L$ (steps)")
ax.set_ylabel(r"$\bar{\Delta}$ (shared $-$ indep.\ captures)")
ax.set_xticks(L_VALS)
ax.set_xticklabels([str(v) for v in L_VALS])
ax.set_xlim(-1, 70)
ax.set_ylim(-1.65, 2.55)
ax.yaxis.grid(True)
ax.set_axisbelow(True)

# Legend — two entries only (sig / n.s. indicated by fill)
slow_patch = mpatches.Patch(color=COL_SLOW,
                             label=r"slow ($T_{\rm corr}{=}33$, $\omega{=}1.5$)")
mid_patch  = mpatches.Patch(color=COL_MID,
                             label=r"mid ($T_{\rm corr}{=}10$, $\omega{=}5.0$)")
sig_note   = mpatches.Patch(facecolor="gray", edgecolor="none",
                             label=r"filled = $p{<}0.05$, open = n.s.")
ax.legend(handles=[slow_patch, mid_patch, sig_note],
          loc="upper right", frameon=False, handlelength=1.2,
          handletextpad=0.5, borderpad=0)

fig.tight_layout()
out = os.path.join(OUTDIR, "fig_crossover.pdf")
fig.savefig(out)
print(f"Saved: {out}")
plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: L_max vs T_corr scaling
# ════════════════════════════════════════════════════════════════════════════
# Data points: (T_corr, L_max) — reliable points for the fit
# fast excluded (no reliable L_max); ω=2.5 shown but excluded from fit
t_corr_pts = np.array([67, 33, 10])
lmax_pts   = np.array([67, 10,  3])

# Force-through-origin fit (3 pts including boundary very_slow)
c_fit = np.sum(t_corr_pts * lmax_pts) / np.sum(t_corr_pts**2)
t_line = np.linspace(0, 75, 200)
lmax_line = c_fit * t_line

# Prediction line (c=1.0)
pred_line = 1.0 * t_line

fig2, ax2 = plt.subplots(figsize=(3.00, 2.35))

# Prediction band (c=1.0, dashed gray)
ax2.plot(t_line, pred_line, color="#999999", lw=1.0, ls="--",
         label=r"$L{=}T_{\rm corr}$ reference")

# Interior-anchor-only line (c=0.30, thin blue dotted)
c_interior = 0.30
ax2.plot(t_line, c_interior * t_line, color="#1A6FBF", lw=1.0, ls=":",
         label=r"interior anchors only: $c{=}0.30$", zorder=1)

# Fitted line (c≈0.85, 3-point fit including boundary)
ax2.plot(t_line, lmax_line, color="#333333", lw=1.4, ls="-",
         label=f"3-point fit (incl. boundary): $c={c_fit:.2f}$")

# Data points — very_slow open (boundary), slow+mid filled (clean anchors)
ax2.scatter([67], [67], color="#999999", marker="s", s=40,
            zorder=5, label=r"very\_slow (boundary)")
ax2.scatter([33, 10], [10, 3], color="#1A6FBF", marker="o", s=44,
            zorder=5, label="slow, mid (interior anchors)")

# ω=2.5: shown as open triangle — no reliable L_max, excluded from fit
# Shown at T_corr=20 with an arrow indicating "ambiguous ↑" direction
ax2.scatter([20], [45], color="#888888", marker="^", s=36,
            facecolors="white", edgecolors="#888888", lw=1.0,
            zorder=5, label=r"$\omega{=}2.5$ (no reliable $L_{\max}$)")
ax2.annotate("", xy=(20, 50), xytext=(20, 46),
             arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

# Labels on data points
for tx, ly, lbl in zip(t_corr_pts, lmax_pts,
                        ["very\\_slow", "slow", "mid"]):
    offset_x = -5 if tx == 67 else 2
    offset_y = 2  if tx == 67 else -4
    ax2.annotate(lbl, (tx, ly),
                 xytext=(tx + offset_x, ly + offset_y),
                 fontsize=6.5, color="#333333", ha="center")
ax2.annotate(r"$\omega{=}2.5$", (20, 45),
             xytext=(22, 43), fontsize=6.5, color="#888888", ha="left")

# 0.30 annotation — label the interior-anchor dotted line
ax2.text(60, c_interior * 60 + 2, r"$c{=}0.30$",
         fontsize=7, color="#1A6FBF", ha="left", va="bottom")

ax2.set_xlabel(r"$T_{\rm corr} = 1/(\omega\cdot dt)$, one-radian decorr.\ timescale (steps)")
ax2.set_ylabel(r"$L_{\max}$ (steps)")
ax2.set_xlim(-2, 75)
ax2.set_ylim(-3, 75)
ax2.set_xticks([0, 10, 20, 33, 45, 67])
ax2.set_yticks([0, 10, 20, 33, 45, 67])
ax2.yaxis.grid(True)
ax2.set_axisbelow(True)
ax2.legend(loc="upper left", frameon=False, fontsize=6.5,
           handlelength=1.2, handletextpad=0.4, borderpad=0)

fig2.tight_layout()
out2 = os.path.join(OUTDIR, "fig_scaling.pdf")
fig2.savefig(out2)
print(f"Saved: {out2}")
plt.close(fig2)

print("Done.")
