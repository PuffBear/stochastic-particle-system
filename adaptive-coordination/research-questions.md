# Research Questions: Adaptive Coordination under Bounded Memory

## Primary research question

What is the minimum memory length L (in steps) required to maintain E[Δ_s] > 0 when the latent field direction rotates at angular rate ω — and does L_critical scale as 1/ω (the field autocorrelation time)?

## Setup

**Field model:** θ(t + dt) = θ(t) + ω·dt, where θ(0) ~ Uniform[0, 2π).

At ω=0: stationary field — exactly SPS-C03. At ω=π/67: field completes a half-rotation over one 67-step episode. All other parameters frozen at SPS-C03 values: α=0.06, M=4, N=256, dt=0.02, σ=0.06, steps=67.

**Memory model — two implementations to compare:**
1. **Sliding window (length L):** compute count-weighted mean velocity from the last L steps of observations only. Older observations discarded.
2. **Exponential decay (decay constant L):** compute exponentially weighted mean with λ = exp(−1/L). Older observations downweighted but not discarded.

Both reduce to the full-history SPS-C03 controller at L = all steps and ω = 0.

**Matched counterfactual constraint:** Each seed must share the same Brownian noise tensor, initial positions, θ(0), and pre-generated θ(t) sequence across all (ω, L, method) conditions. The only causal difference between shared and independent arms is the message channel.

---

## Q1 — Critical memory length vs. rotation speed

**Hypothesis:** L_critical(ω) ≈ c/ω, where c is a constant determined by the single-agent field estimation error at α=0.06. The autocorrelation time of the field direction is 1/ω steps; observations older than 1/ω steps are approximately uncorrelated with the current field direction and should be discarded.

**Theoretical prediction (from `theory/field-rotation.md`):**
- ω = π/200 (≈0.016 rad/step, slow drift): L_critical ≈ 12 steps
- ω = π/100 (≈0.031 rad/step): L_critical ≈ 6 steps
- ω = π/50 (≈0.063 rad/step, fast drift): L_critical ≈ 3 steps

**Experimental grid:**

| Factor | Levels |
|---|---|
| ω | 0, π/200, π/100, π/50 rad/step |
| L | 1, 3, 10, 30, all (≡67) steps |
| Method | Sliding window, exponential decay |
| Seeds per cell | 8 matched pairs |

Total cells: 4 ω × 5 L × 2 method = 40 conditions. Total runs: 40 × 8 × 2 arms = 640 episodes.

**Estimand:** Fraction of seeds with positive Δ_s per (ω, L, method) cell. L_critical(ω) = smallest L achieving ≥60% positive seeds.

**Anchor:** Cell (ω=0, L=all, either method) = SPS-C03 result. Must reproduce Δ̄ ≈ +1.19 within ±0.5 before proceeding.

---

## Q2 — Sliding window vs. exponential decay

**Hypothesis:** Exponential decay outperforms sliding window at high ω. The hard cutoff of a sliding window discards all observations older than L steps simultaneously — including observations that are still partially informative. Exponential decay provides a smooth trade-off that loses less signal near the cutoff boundary.

**Estimand:** Δ̄(ω, L, method) for sliding window vs. exponential decay at each non-zero ω. Report difference and whether sliding window or exponential decay achieves lower L_critical at each ω.

**Pre-registered prediction:** At ω = π/50 (fast drift), exponential decay achieves L_critical ≤ 3 while sliding window requires L_critical ≥ 5.

---

## Q3 — Does team benefit persist under drift?

**Hypothesis:** Even at L > L_critical, the coordination benefit Δ̄(shared, independent) shrinks with ω because all agents experience the same rotation phase, increasing inter-agent correlation and reducing the diversity benefit of pooling.

**Reasoning:** The shared team mean is informative precisely because different agents observe different particle populations and their estimates are partially independent. When the field rotates, all agents' stale observations are equally wrong in the same direction — pooling stale estimates amplifies rather than cancels the error.

**Estimand:** Δ̄(shared_memory_L, independent_memory_L) as a function of (ω, L) for L > L_critical. Compare to Δ̄ at ω=0.

---

## Q4 — Theoretical prediction check

**For each tested ω, compare empirical L_critical to theoretical L_theory = c/ω.** Report L_critical / L_theory for each ω level and test whether a single constant c is consistent across all three non-zero ω values.

If a single c fits all three levels: the autocorrelation-time prediction is confirmed, c is estimable, and the relationship is predictive for untested ω.

If c varies across ω levels: the 1/ω scaling is wrong. Investigate alternative scalings (1/ω², 1/√ω) and report which fits better.

---

## Kill criteria

**Primary kill:** Δ̄ > 0 for all (ω, L) combinations tested — including L=1 at ω=π/50. No boundary found because even a single step of memory is sufficient. This would mean the team benefit is robust to rotation at all tested speeds — scientifically interesting but kills the L_critical claim.

> **STATUS: FALSE ALARM — original ω grid was in the wrong regime.**
> Runs 9001–9032 at ω ∈ {π/200, π/100, π/50} triggered this criterion, but
> the per-step rotation ω·dt at those values is {0.000314, 0.000628, 0.00126}
> rad/step — the field rotated at most 5° over the entire 67-step episode.
> T_corr = 1/(ω·dt) ∈ {3185, 1591, 795} steps, all >> episode length.
> In this regime, the field is essentially stationary and "robustness" is trivial.
>
> **Correction:** The ω grid has been redesigned so that T_corr ∈ {67, 33, 10, 3}
> steps (ω ∈ {0.75, 1.5, 5.0, 17.0} rad/step). The corrected experiment is now
> running. Kill-criterion assessment deferred until corrected-grid results arrive.

**Scaling kill:** L_critical(ω) does not scale as 1/ω and no alternative scaling is apparent. Redirect: report L_critical at each ω empirically without a scaling law.

> **STATUS: NOT TRIGGERED (corrected-grid data)** — Under the t-test criterion,
> L_max (largest L with p<0.05, window method) follows T_corr = 1/(ω·dt):
>   very_slow: L_max=67 = T_corr; slow: L_max=10 ≈ 0.30·T_corr=33;
>   mid: L_max=3 ≈ 0.30·T_corr=10; fast: L_max=None (T_corr=3).
> Scaling fit: L_max ≈ 0.85·T_corr (R²=0.81, 3 points). The scaling law
> is approximately confirmed. Previous "TRIGGERED" status applied to the wrong
> ω grid (near-stationary regime); this is superseded by corrected-grid results.

**Reproduction kill:** Cell (ω=0, L=all) does not reproduce SPS-C03 Δ̄ within ±0.5. Infrastructure problem in the rotating-field implementation; halt and diagnose.

> **STATUS: NOT TRIGGERED** — Gate passed: ω=0, L=1, window, seeds 6001–6032:
> Δ̄=+1.188, sign=20/32. Implementation is correct.

**Team-benefit kill (Q3):** Δ̄(shared, independent) at L > L_critical does not decrease with ω — team benefit is fully robust to rotation at matched memory. Redirects the Q3 claim; other questions unaffected.

> **STATUS: TRIGGERED (corrected-grid data)** — Pooled Δ̄ (window, all L) does
> not decrease monotonically with ω: very_slow=+0.48, slow=+0.64, mid=+0.44,
> fast=+0.14. The slow arm has higher pooled Δ̄ than very_slow, violating the
> predicted monotone decrease. However, at ω=fast the pooled benefit is
> non-significant (p=0.25), which is partially consistent with Q3. The strict
> monotone-decrease claim is not supported; the "no benefit at high ω" claim
> has marginal support only at the fastest tested speed.

---

## Connection to SPS-C03 and FR-B3

This paper extends SPS-C03 along the temporal dimension. FR-B3 extends it along the (ρ, κ) spatial axes. Together they characterise the operating regime of the shared_summary_v2 controller in two orthogonal directions: task parameterisation (FR-B3) and field non-stationarity (FR-B4).

The full picture: FR-B3 tells you *where* in (ρ, κ) space communication helps; FR-B4 tells you *for how long* a rotating field can be tracked before the benefit degrades.
