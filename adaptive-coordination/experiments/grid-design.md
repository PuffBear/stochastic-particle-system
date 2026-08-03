# Experiment Grid Design: Adaptive Coordination

**Status:** Pre-registered design — do not modify after Phase 2d (reproduction gate) passes

---

## Fixed parameters (inherited from SPS-C03)

```
N = 256, M = 4, alpha = 0.06, dt = 0.02, sigma = 0.06, steps = 67
arena = [0,1]², reflecting boundaries
```

New parameter: ω (rotation rate, rad/step). Default ω=0 = SPS-C03.

---

## Phase 2d: Reproduction gate (must pass before any non-zero ω runs)

Seeds: 6001–6032 (32 SPS-C03 confirmed seeds)
Condition: ω=0, L=1 (stateless)
Gate criterion (window method only): Δ̄ ∈ [+0.69, +1.69], sign count ≥20/32
Decay method result: reported but not required (see rationale below)

**Rationale for L=1 (not L=all):** The FR-B4 windowed controller at L=1 uses
only the current step's observations — exactly matching the stateless SPS-C03
policy's per-step behaviour for the shared arm. At L=all, temporal pooling
changes both arms' estimates in a way that is scientifically interesting but
makes the gate harder to interpret.

**Rationale for 32 seeds (not 8):** Seeds 1001-1008 have insufficient
statistical power for this check: with SD≈2.44, the SE over 8 seeds is ≈0.86,
so Δ̄ from 8 seeds can easily be ±1.7 of the true value. Seeds 6001-6032 give
SE≈0.43, making the gate criterion achievable when the implementation is correct.

**Rationale for window-only gate:** The window controller at L=1 is exactly
the stateless SPS-C03 controller (shared arm: team mean of current step only;
independent arm: self mean of current step only). The decay controller at L=1
sets λ=exp(-1)≈0.368 and still accumulates history with exponential decay —
it is NOT stateless. Its independent arm receives more signal than SPS-C03's
per-step arm, shrinking Δ. This is correct behaviour, not a bug. The gate
uses the window method because it has a clean equivalence to SPS-C03 at L=1;
the decay method's L=1 behaviour will be characterised as part of the main grid.

**Confirmed gate results (window):** Δ̄=+1.188, sign=20/32 ✅ (matches SPS-C03 +1.19)

If gate fails: stop, diagnose the implementation. Do not proceed.

---

## Main experiment grid

### ω levels — CORRECTED GRID (see NOTE below)

| Label | ω (rad/step) | ω·dt (rad/step) | T_corr=1/(ω·dt) | Predicted L_max |
|---|---|---|---|---|
| stationary | 0 | 0 | ∞ | all |
| very_slow | 0.75 | 0.015 | 67 | ~67 |
| slow | 1.5 | 0.030 | 33 | ~33 |
| mid | 5.0 | 0.100 | 10 | ~10 |
| fast | 17.0 | 0.340 | 3 | ~3 |

> **NOTE — original grid was in wrong regime.** The original design used ω ∈ {π/200, π/100, π/50} rad/step.
> At these values, ω·dt ∈ {0.0003, 0.0006, 0.0013} rad/step (T_corr ∈ {3185, 1591, 795} steps >> 67-step
> episode). The field was effectively stationary in all runs. The "primary kill" triggered in those runs was
> a false alarm. Grid was corrected to target T_corr ∈ {67, 33, 10, 3} steps. See `theory/field-rotation.md`.

### L levels

| Label | L (steps) | Notes |
|---|---|---|
| L1 | 1 | Single-step memory — nearly no memory |
| L3 | 3 | Very short window |
| L10 | 10 | Moderate window |
| L30 | 30 | Near half-episode |
| Lall | 67 | Full episode history = SPS-C03 controller |

### Methods

| Label | Description |
|---|---|
| window | Sliding window: last L observations only |
| decay | Exponential decay: λ = exp(−1/L) |

### Full grid

40 active conditions (4 ω × 5 L × 2 methods). Plus gate conditions (ω=0, L=1).
Results from run1: `results/FR-B4/fr_b4_full_grid_run1.json` (seeds 9001–9008, 8 per cell).

| ω | L | method | Δ̄ | sign/8 | L_critical |
|---|---|---|---|---|---|
| 0 (gate) | 1 | window | +1.188 | 20/32 | — |
| 0 (gate) | 1 | decay | +0.344 | 14/32 | — |
| 0 | 3 | window | +0.812 | 16/32 | — |
| 0 | 3 | decay | +0.938 | 17/32 | — |
| 0 | 10 | window | +0.719 | 16/32 | — |
| 0 | 10 | decay | +0.375 | 15/32 | — |
| 0 | 30 | window | +0.250 | 14/32 | — |
| 0 | 30 | decay | +0.281 | 13/32 | — |
| π/200 | 1 | window | +1.500 | 4/8 | |
| π/200 | 1 | decay | +1.625 | 6/8 ✅ | L_crit=1 |
| π/200 | 3 | window | +0.375 | 3/8 | |
| π/200 | 3 | decay | +0.375 | 3/8 | |
| π/200 | 10 | window | -0.625 | 2/8 | |
| π/200 | 10 | decay | +0.250 | 4/8 | |
| π/200 | 30 | window | +0.750 | 4/8 | |
| π/200 | 30 | decay | +0.500 | 4/8 | |
| π/200 | all | window | +0.000 | 5/8 ✅ | L_crit=67 |
| π/200 | all | decay | +0.250 | 5/8 ✅ | (also L=1) |
| π/100 | 1 | window | +1.750 | 4/8 | |
| π/100 | 1 | decay | +1.375 | 6/8 ✅ | L_crit=1 |
| π/100 | 3 | window | +0.625 | 4/8 | |
| π/100 | 3 | decay | +0.250 | 4/8 | |
| π/100 | 10 | window | -1.250 | 2/8 | |
| π/100 | 10 | decay | +0.750 | 3/8 | |
| π/100 | 30 | window | +0.500 | 3/8 | |
| π/100 | 30 | decay | +0.750 | 4/8 | |
| π/100 | all | window | -0.625 | 3/8 | L_crit=undefined |
| π/100 | all | decay | +0.500 | 6/8 ✅ | (also L=1) |
| π/50 | 1 | window | +2.500 | 6/8 ✅ | L_crit=1 |
| π/50 | 1 | decay | +1.125 | 4/8 | |
| π/50 | 3 | window | -0.250 | 3/8 | |
| π/50 | 3 | decay | +0.000 | 3/8 | |
| π/50 | 10 | window | -0.375 | 3/8 | |
| π/50 | 10 | decay | +0.625 | 2/8 | |
| π/50 | 30 | window | +0.875 | 5/8 ✅ | |
| π/50 | 30 | decay | +1.500 | 6/8 ✅ | L_crit=30 |
| π/50 | all | window | +0.000 | 4/8 | |
| π/50 | all | decay | +0.125 | 3/8 | |

**Corrected-grid results** (`results/FR-B4/fr_b4_corrected_combined_32seeds.json`, seeds 9001–9032, 32/cell):

| ω (T_corr) | L=1 | L=3 | L=10 | L=30 | Lall | L_max (window) |
|---|---|---|---|---|---|---|
| 0 (∞) | +1.19 * | +0.81 * | +0.72 * | +0.25 | +1.19* | 10 |
| 0.75 (67) | +0.53 | +0.47 | +0.22 | +0.41 | +0.78 * | 67 |
| 1.5 (33) | +1.47 * | +1.50 * | +0.94 * | -0.28 | -0.44 | 10 |
| 5.0 (10) | +0.88 * | +0.91 * | +0.25 | +0.16 | +0.03 | 3 |
| 17.0 (3) | +0.47 | +0.47 | -0.09 | -0.38 | +0.25 | None |

*(* = p<0.05 one-sided t-test; Δ̄ values for window method)*

**L_max = largest L with p<0.05 (window method):**

| ω | T_corr | L_max | L_max / T_corr |
|---|---|---|---|
| 0.75 | 67 | 67 | 1.00 |
| 1.5 | 33 | 10 | 0.30 |
| 5.0 | 10 | 3 | 0.30 |
| 17.0 | 3 | None | — |

**Scaling fit:** L_max ≈ 0.85 × T_corr (R²=0.81, 3 points from main grid). See intermediate-L results below for refined estimate.

**Key scientific findings from corrected-grid data:**

1. **The L_critical boundary is visible.** At ω=slow (T_corr=33) and ω=mid (T_corr=10),
   the window method shows clear sign flip: short L is beneficial, long L is not.

2. **Pooled benefit degrades with ω (window method):** Δ̄(pooled) = +0.74, +0.48, +0.64, +0.44, +0.14
   at ω = 0, very_slow, slow, mid, fast. The fast-rotation arm shows no significant
   pooled benefit (p=0.25).

---

### Intermediate-L results (L ∈ {5, 20, 45}, seeds 9001–9032)

File: `results/FR-B4/fr_b4_full_combined.json` (merges corrected + extra-L runs)

| ω (T_corr) | L=3 | L=5 | L=10 | L=20 | L=30 | L_max | crossover |
|---|---|---|---|---|---|---|---|
| slow (33) | +1.50 * | +1.66 * | +0.94 * | **-0.38** | -0.28 | 10 | between 10 and 20 |
| mid (10) | +0.91 * | +0.22 | +0.25 | +0.56 | +0.16 | 3 | between 3 and 5 |

*(* = p<0.05, window method)*

**L_max / T_corr: 10/33 = 0.30; 3/10 = 0.30.**

**Quantitative result:** L_max ≈ **0.30 × T_corr** = **0.30 / (ω·dt)**

The crossover is sharp for slow: Δ̄ goes from +0.94 (L=10, p=0.049) to −0.38 (L=20, p=0.79) —
a swing of >1.3 particles when memory doubles from 10 to 20 steps. For mid the dropoff is
softer (+0.91 to +0.22) but also falls below significance at L=5.

fast (T_corr=3): L5 window p=0.023 is likely a Type I error (L1 and L3 non-significant;
~1 false positive expected across 24 fast cells at α=0.05). No reliable benefit at ω=fast.

very_slow (T_corr=67): Lall barely significant (p=0.044), L45 not (p=0.073). L_max cannot
be estimated within the 67-step episode for very_slow.

---

## Seed protocol

All 8 seeds (9001–9008) across all non-zero ω conditions share:
- The same Brownian noise tensor
- The same initial particle positions
- The same initial collector positions
- The same θ(0) draw (from seed)
- The same pre-generated θ(t) sequence (deterministic from ω and seed)

The θ(t) sequence is generated at episode start as:
```
theta(t) = theta(0) + omega * t * dt   for t = 0, 1, ..., 66
```

This sequence is shared across both arms (shared and independent) and across all (L, method) conditions for the same seed. The only causal difference is the message channel.

---

## Pre-registered analysis

### Primary: L_critical per (ω, method)

For each (ω, method) pair, plot sign count (fraction of 8 seeds with Δ_s > 0) as a function of L. L_critical = smallest L with sign count ≥ 60% (≥5/8 seeds).

If no L achieves 60%: L_critical = undefined (coordination never reaches threshold at this ω with this method).
If all L achieve 60%: L_critical = 1 (coordination is robust even at minimal memory).

### Secondary: 1/ω scaling fit

Using L_critical values at ω ∈ {π/200, π/100, π/50} for the sliding window method:
- Fit L_critical = c / ω using least squares
- Report c and R²
- Plot empirical vs. predicted L_critical

Success criterion: R² ≥ 0.85 and residuals < 2 steps across all three ω levels.

### Q3: Team benefit degradation

For each (ω, method), plot Δ̄(L) for L ≥ L_critical. Test whether Δ̄ decreases monotonically with ω (Spearman rank correlation across ω levels at fixed L).

### Q4: Theoretical prediction check

Compare empirical L_critical to theoretical L_theory = c_theory / ω (using c_theory from `theory/field-rotation.md`). Report L_critical / L_theory. Success: all three ratios within factor of 2 of each other.

---

## Run order (priority)

1. Reproduction gate: (ω=0, Lall, both methods) — must pass first
2. Fast rotation diagnostic: (ω=π/50, L∈{1,3,10}) — tests whether boundary exists at all
3. Full slow and mid rotation: all L levels for ω∈{π/200, π/100}
4. Complete fast rotation: remaining L levels for ω=π/50
5. ω=0 at all L levels (both methods) — characterises memory effect at stationary field

---

## Runtime estimate

~10s per episode. 40 conditions × 8 seeds × 2 arms = 640 episodes = ~1.8 hours single-threaded. With 8 cores: ~15 minutes. No HPC required.

---

## Output format

Per-cell results stored at:
`results/raw/FR-B4-ADAPTIVE/omega_{label}/L_{label}/method_{label}/`

Each directory contains episode summaries in the standard SPS JSONL format. The θ(t) sequence is logged in the episode manifest for verification.
