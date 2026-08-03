"""FR-B4 result analysis: per-cell t-tests, per-omega pooled estimates, L_critical.

Usage:
    python analysis/analyze_fr_b4_results.py \
        results/FR-B4/fr_b4_combined_32seeds.json

Outputs:
  - Per-cell table: Δ̄, SD, SE, 95% CI, t-stat, p-value (one-sided H0: Δ≤0)
  - Per-(omega,method) pooled Δ̄ across L levels
  - L_critical estimates with sign-count criterion and t-test criterion
  - 1/omega scaling fit (if ≥2 omega levels have defined L_critical)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


OMEGA_LABELS  = ["stationary", "very_slow", "slow", "mid", "fast"]
OMEGA_VALUES  = {"stationary": 0.0, "very_slow": 0.75, "slow": 1.5, "mid": 5.0, "fast": 17.0}
L_LABELS      = ["L1", "L3", "L5", "L10", "L20", "L30", "L45", "Lall"]
L_STEPS       = {"L1": 1, "L3": 3, "L5": 5, "L10": 10, "L20": 20, "L30": 30, "L45": 45, "Lall": 67}
METHODS       = ["window", "decay"]
LCRIT_FRAC    = 0.60
T_CRIT_ALPHA  = 0.05  # one-sided


def _t_pvalue(t: float, df: int) -> float:
    """Approximate one-sided p-value P(T > t) using normal approximation (df≥10)."""
    if df <= 0:
        return float("nan")
    # For df large enough, t ~ N(0,1); use erfc approximation
    x = abs(t) / math.sqrt(2)
    p_two = math.erfc(x)
    p_one = p_two / 2
    return p_one if t > 0 else 1.0 - p_one


def _cell_analysis(cell: dict[str, Any]) -> dict[str, Any]:
    deltas = [r["delta"] for r in cell["per_seed"]]
    n  = len(deltas)
    db = cell["delta_bar"]
    sd = cell["delta_sd"]
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t  = db / se if se > 0 else float("nan")
    df = n - 1
    p  = _t_pvalue(t, df)
    ci_lo = db - 1.96 * se
    ci_hi = db + 1.96 * se
    sc = cell["sign_count"]
    return {
        "n": n, "delta_bar": db, "sd": sd, "se": se,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "t": t, "p_one": p, "sign_count": sc,
        "sign_frac": sc / n if n > 0 else float("nan"),
        "sign_pass": (sc / n >= LCRIT_FRAC) if n > 0 else False,
        "t_pass": (math.isfinite(p) and p < T_CRIT_ALPHA),
    }


def main() -> None:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())

    # Index cells — accepts both raw run JSON ("cells") and combined JSON ("results")
    cells_raw: dict[str, dict[str, Any]] = {}
    for entry in data.get("cells", data.get("results", [])):
        if entry.get("gate_cell"):
            continue
        key = f"{entry['omega_label']}|{entry['L_label']}|{entry['method']}"
        cells_raw[key] = entry

    # Per-cell analysis
    cells: dict[str, dict[str, Any]] = {k: _cell_analysis(v) for k, v in cells_raw.items()}

    # ── Per-cell table ────────────────────────────────────────────────────────
    print("=== Per-cell results ===")
    print(f"{'omega':12s} {'L':5s} {'method':6s}  {'n':>3s}  {'Δ̄':>6s}  {'SE':>4s}  {'95% CI':>15s}  {'p':>6s}  sign/n  pass")
    for omega_label in OMEGA_LABELS:
        for L_label in L_LABELS:
            for method in METHODS:
                key = f"{omega_label}|{L_label}|{method}"
                c = cells.get(key)
                if c is None:
                    continue
                ci = f"[{c['ci_lo']:+.2f},{c['ci_hi']:+.2f}]"
                p_str = f"{c['p_one']:.3f}" if math.isfinite(c["p_one"]) else "  NaN"
                flag = " *" if c["t_pass"] else ("  ·" if c["sign_pass"] else "")
                print(
                    f"  {omega_label:12s} {L_label:5s} {method:6s}  {c['n']:3d}  "
                    f"{c['delta_bar']:+6.3f}  {c['se']:4.3f}  {ci:>15s}  {p_str}  "
                    f"{c['sign_count']}/{c['n']}{flag}"
                )

    # ── Per-(omega, method) pooled across L ──────────────────────────────────
    print("\n=== Pooled per-(omega, method): all L levels combined ===")
    print(f"{'omega':12s} {'method':6s}  {'n':>4s}  {'Δ̄':>6s}  {'SE':>4s}  {'95% CI':>15s}  {'p':>6s}")
    for omega_label in OMEGA_LABELS:
        for method in METHODS:
            all_deltas: list[float] = []
            for L_label in L_LABELS:
                key = f"{omega_label}|{L_label}|{method}"
                raw = cells_raw.get(key)
                if raw is None:
                    continue
                all_deltas.extend(r["delta"] for r in raw["per_seed"])
            if not all_deltas:
                continue
            n  = len(all_deltas)
            db = sum(all_deltas) / n
            var = sum((d - db)**2 for d in all_deltas) / (n - 1)
            sd = math.sqrt(var)
            se = sd / math.sqrt(n)
            t  = db / se
            p  = _t_pvalue(t, n - 1)
            ci = f"[{db-1.96*se:+.2f},{db+1.96*se:+.2f}]"
            p_str = f"{p:.4f}" if math.isfinite(p) else "   NaN"
            print(f"  {omega_label:12s} {method:6s}  {n:4d}  {db:+6.3f}  {se:4.3f}  {ci:>15s}  {p_str}")

    # ── L_critical (two criteria) ─────────────────────────────────────────────
    # L_critical = smallest L passing; L_max = largest L passing (last before degradation)
    print("\n=== L_critical summary (min and max L passing t-test p<0.05) ===")
    print(f"{'omega':12s} {'method':6s}  {'T_corr':>6s}  {'L_min':>5s}  {'L_max':>5s}  (L_max=last significant L)")
    DT = 0.02
    l_max_t: dict[str, int | None] = {}
    l_crit_t: dict[str, int | None] = {}
    for omega_label in OMEGA_LABELS:
        for method in METHODS:
            t_lcrit = t_lmax = None
            for L_label in L_LABELS:
                key = f"{omega_label}|{L_label}|{method}"
                c = cells.get(key)
                if c is None:
                    continue
                if t_lcrit is None and c["t_pass"]:
                    t_lcrit = L_STEPS[L_label]
                if c["t_pass"]:
                    t_lmax = L_STEPS[L_label]
            l_crit_t[f"{omega_label}|{method}"] = t_lcrit
            l_max_t[f"{omega_label}|{method}"]  = t_lmax
            omega_val = OMEGA_VALUES[omega_label]
            t_corr = f"{1/(omega_val*DT):.0f}" if omega_val > 0 else "∞"
            print(f"  {omega_label:12s} {method:6s}  {t_corr:>6s}  {str(t_lcrit):>5s}  {str(t_lmax):>5s}")

    # ── 1/(omega*dt) fit on L_max ─────────────────────────────────────────────
    print("\n=== L_max scaling: L_max (last significant L) vs T_corr = 1/(omega*dt) ===")
    for method in METHODS:
        points = []
        for omega_label in ["very_slow", "slow", "mid", "fast"]:
            omega = OMEGA_VALUES[omega_label]
            t_corr = 1.0 / (omega * DT)
            lm = l_max_t.get(f"{omega_label}|{method}")
            if lm is not None:
                points.append((t_corr, lm))
                print(f"  {method}  {omega_label:10s}  T_corr={t_corr:.0f}  L_max={lm}")
            else:
                print(f"  {method}  {omega_label:10s}  T_corr={t_corr:.0f}  L_max=None (no significant L)")
        if len(points) >= 2:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            # force-through-origin fit: c = sum(x*y) / sum(x^2)
            c_fit = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
            ss_res = sum((y - c_fit * x) ** 2 for x, y in zip(xs, ys))
            ss_tot = sum((y - sum(ys)/len(ys)) ** 2 for y in ys)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"  {method}: L_max ≈ {c_fit:.3f} * T_corr  R²={r2:.3f}  (from {len(points)} omega levels)")
        else:
            print(f"  {method}: fewer than 2 defined L_max values — no fit")


if __name__ == "__main__":
    main()
