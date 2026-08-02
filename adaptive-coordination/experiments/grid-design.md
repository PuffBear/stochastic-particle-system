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

Seeds: 1001–1008 (first 8 SPS-C03 seeds)
Condition: ω=0, L=all, both methods
Gate: Δ̄ ∈ [+0.69, +1.69], sign count ≥5/8

If gate fails: stop, diagnose the rotating-field implementation. Do not proceed.

---

## Main experiment grid

### ω levels

| Label | ω (rad/step) | Period (steps) | T_corr (steps) | Predicted L_critical |
|---|---|---|---|---|
| stationary | 0 | ∞ | ∞ | all |
| slow | π/200 ≈ 0.0157 | 400 | 64 | ~30 |
| mid | π/100 ≈ 0.0314 | 200 | 32 | ~10 |
| fast | π/50 ≈ 0.0628 | 100 | 16 | ~3 |

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

40 active conditions (4 ω × 5 L × 2 methods). Plus 2 anchor conditions (ω=0 × Lall × both methods).

| ω | L | method | Seeds | Status |
|---|---|---|---|---|
| 0 | all | window | 1001–1008 | ✅ Anchor (C03) |
| 0 | all | decay | 1001–1008 | ✅ Anchor (must match) |
| π/200 | 1,3,10,30,all | window | 9001–9008 | To run |
| π/200 | 1,3,10,30,all | decay | 9001–9008 | To run |
| π/100 | 1,3,10,30,all | window | 9001–9008 | To run |
| π/100 | 1,3,10,30,all | decay | 9001–9008 | To run |
| π/50 | 1,3,10,30,all | window | 9001–9008 | To run |
| π/50 | 1,3,10,30,all | decay | 9001–9008 | To run |

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
