# Research Questions: Adaptive Coordination under Bounded Memory

## Primary research question

What is the minimum memory length L (in steps) required to maintain E[Δ_s] > 0 when the latent field orientation rotates at angular rate ω — and does L_critical scale as 1/ω (the field autocorrelation time)?

## Setup

**Field model:** θ(t + dt) = θ(t) + ω · dt. At ω=0: stationary SPS-C03 baseline. At ω=π/67: half-rotation over one episode.

**Memory model — two implementations:**
1. **Sliding window:** average of last L step observations, then Proposition 2 count-weighting
2. **Exponential decay:** exponentially weighted mean with λ = exp(−1/L)

All other parameters frozen at SPS-C03 values: α=0.06, M=4, N=256, dt=0.02, 67 steps.

## Sub-questions

### Q1 — Critical memory length vs. rotation speed

**Hypothesis:** L_critical(ω) ≈ c/ω, where c ≈ 12 steps at α=0.06 (estimated from single-agent field estimation error). Concretely:
- ω = π/200 (slow drift): L_critical ≈ 8 steps
- ω = π/50 (fast drift): L_critical ≈ 2 steps

**Estimand:** Fraction of seeds with positive contrast per (ω, L) cell. L_critical(ω) = smallest L achieving ≥60% positive seeds.

**Conditions:**

| Factor | Levels |
|---|---|
| ω (rotation) | 0, π/200, π/100, π/50 rad/step |
| L (memory) | 1, 3, 10, 30, all steps since episode start |
| Method | Sliding window, exponential decay |
| Seeds per cell | 8 matched pairs |

### Q2 — Sliding window vs. exponential decay

**Hypothesis:** Exponential decay outperforms sliding window at high ω. The hard cutoff of a sliding window discards informative observations abruptly; exponential decay provides a smooth trade-off.

**Estimand:** Δ̄(ω, L, method) for both methods at the three nonzero rotation speeds.

### Q3 — Does team benefit persist under drift?

**Hypothesis:** Even above L_critical, the coordination benefit shrinks with ω because all agents experience the same rotation phase, increasing inter-agent correlation and reducing the diversity benefit of pooling.

**Estimand:** Δ̄(shared_memory_L, independent_memory_L) as a function of (ω, L).

### Q4 — Theoretical prediction check

For each tested ω, compare empirical L_critical to theoretical L_theory = c/ω. Report L_critical / L_theory and test whether a single constant c is consistent across all ω levels.

## Kill criteria

- **Primary kill:** Δ̄ > 0 for all (ω, L) combinations with L ≥ 3 at all tested rotation speeds. No boundary found.
- **Scaling kill:** L_critical(ω) does not scale as 1/ω. The autocorrelation-time prediction is wrong and no alternative scaling is apparent.

## Connection to SPS baseline

The (ω=0, L=all-steps) condition is exactly SPS-C03. Memory effects are measured as degradation in Δ̄ relative to ω=0, controlling for the direct effect of a rotating field on captures independent of communication.
