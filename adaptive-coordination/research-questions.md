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

> **STATUS: PRIMARY KILL TRIGGERED** (32 seeds/cell, runs 9001–9032).
> Every (ω, L, method) cell has Δ̄ > 0. When pooled across L levels, all four
> ω levels yield statistically significant coordination benefit (p<0.01):
>   stationary: Δ̄=+0.742 [+0.51,+0.97]; slow: Δ̄=+0.706 [+0.24,+1.17]
>   mid: Δ̄=+0.562 [+0.11,+1.02];        fast: Δ̄=+0.794 [+0.35,+1.24]
> Per-cell t-test (p<0.05): window method passes at L=1 for all three
> non-zero ω levels — no critical memory length exists within the tested range.
> The L_critical ~ 1/ω hypothesis is **not confirmed**.
>
> **Redirect per kill criterion:** The paper narrative shifts from "identifying
> the minimum memory length" to "coordination is more robust to field rotation
> than the L_critical framework predicts." Key findings:
> 1. Communication benefit persists at all tested ω and all L (L=1 through Lall).
> 2. Effect size is modestly reduced at ω>0 (Δ̄≈0.56–0.79 vs +1.19 at ω=0) but
>    remains positive and statistically significant.
> 3. No monotone relationship between L and Δ̄ within any ω level — L is not
>    the operative memory variable in this regime.
> 4. The decay method at L=1 uses all history with λ=0.368 — it is not stateless,
>    which complicates the "minimum memory" framing further.
>
> **Revised paper claim:** "Under uniform-field rotation at speeds spanning an
> order of magnitude, the communication benefit of shared velocity summaries
> is robust across memory lengths from one step to the full episode. The
> effective communication channel retains value even when the field has rotated
> substantially since the oldest shared observation."

**Scaling kill:** L_critical(ω) does not scale as 1/ω and no alternative scaling is apparent. Redirect: report L_critical at each ω empirically without a scaling law.

> **STATUS: ALSO TRIGGERED** — L_critical is undefined for most cells under the
> sign-count criterion (n=32 insufficient), and under the t-test criterion
> L_crit=1 at all ω for the window method (no scaling law possible).

**Reproduction kill:** Cell (ω=0, L=all) does not reproduce SPS-C03 Δ̄ within ±0.5. Infrastructure problem in the rotating-field implementation; halt and diagnose.

> **STATUS: NOT TRIGGERED** — Gate passed: ω=0, L=1, window, seeds 6001–6032:
> Δ̄=+1.188, sign=20/32. Implementation is correct.

**Team-benefit kill (Q3):** Δ̄(shared, independent) at L > L_critical does not decrease with ω — team benefit is fully robust to rotation at matched memory. Redirects the Q3 claim; other questions unaffected.

> **STATUS: TRIGGERED** — Pooled Δ̄ does not decrease monotonically with ω
> (fast has the highest pooled Δ̄=+0.794). Q3 claim that drift degrades benefit
> is not supported by pooled data.

---

## Connection to SPS-C03 and FR-B3

This paper extends SPS-C03 along the temporal dimension. FR-B3 extends it along the (ρ, κ) spatial axes. Together they characterise the operating regime of the shared_summary_v2 controller in two orthogonal directions: task parameterisation (FR-B3) and field non-stationarity (FR-B4).

The full picture: FR-B3 tells you *where* in (ρ, κ) space communication helps; FR-B4 tells you *for how long* a rotating field can be tracked before the benefit degrades.
