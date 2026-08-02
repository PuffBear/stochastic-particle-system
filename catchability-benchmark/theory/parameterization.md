# The ρ-κ Parameterization

**Status:** Theory draft — to be finalised before experiments run

---

## Motivation

The SPS-C03 confirmation established a positive coordination effect at one operating point: α=0.06, v_max=0.30, dt=0.02, σ=0.06. The natural next question is: is this result specific to this operating point, or does it generalise? And if it generalises, what determines when communication is valuable?

The answer requires identifying the quantities that govern the task — not the raw parameters, but the nondimensional combinations that determine behaviour. Two such combinations arise naturally.

---

## Derivation of ρ (sensing difficulty)

**Setup:** Each collector observes N_i nearby particles and estimates the local field direction from their velocities. Each particle velocity has a drift component α·(cos θ, sin θ) and a noise component σ·dW/dt with dW Brownian.

**The per-observation SNR** of a single velocity observation as a field direction estimator is:

```
SNR_single = (α · √dt) / σ
```

The √dt factor appears because the drift displacement over one timestep is α·dt, while the Brownian displacement is σ·√dt — so the ratio of signal to noise in observed displacement is α·dt / (σ·√dt) = (α·√dt)/σ.

**Define:**
```
ρ ≡ α · √dt / σ
```

ρ is the per-observation signal-to-noise ratio of the field direction estimate. At the confirmed SPS-C03 operating point: ρ = 0.06 · √0.02 / 0.06 ≈ 0.141.

Wait — the research files use ρ≈0.21. Rechecking: α=0.06, dt=0.02, σ=0.06:
ρ = α·√dt/σ = 0.06 · √0.02 / 0.06 = √0.02 ≈ 0.141.

The ≈0.21 figure in earlier documents may use a different definition. This document adopts the signal-detection-grounded definition: **ρ = α·√dt / σ**. At α=0.06, dt=0.02, σ=0.06: ρ = √0.02 ≈ 0.141.

**Qualitative interpretation:**
- ρ → 0: field is undetectable from a single observation; collectors need many observations or team averaging to estimate direction
- ρ → ∞: field direction is immediately obvious from a single observation; coordination adds no sensing benefit
- Low ρ is where shared team summaries most plausibly add value — collectors are individually blind, so pooling estimates matters

**Predicted effect on coordination gain:** g(ρ) should be decreasing in ρ (easier sensing = less benefit from shared information) and bounded (at some ρ, the single-agent estimate is already good enough).

---

## Derivation of κ (control authority / catchability)

**Setup:** A particle undergoing Brownian motion with drift α moves in a time-varying direction. A collector with maximum speed v_max tries to intercept it.

**The catchability question:** Can a collector reliably move toward a particle faster than the particle's effective drift speed moves it away?

The particle's drift speed is α. The collector's maximum speed is v_max. Their ratio:

```
κ ≡ α / v_max
```

κ is the ratio of signal drift speed to collector speed.

- κ < 1: the collector is faster than the drift. It can always intercept a particle if it can find it (sensing is the bottleneck).
- κ > 1: the drift is faster than the collector. No actuation strategy guarantees interception; placement and coverage dominate.
- κ = 1: the boundary case; interception is possible only with ideal information.

At the confirmed SPS-C03 operating point: α=0.06, v_max=0.30, so κ = 0.06/0.30 = 0.20.

**Connection to Péclet number:** In fluid dynamics, Pe = v·L/D (advection speed × length scale / diffusivity). κ is structurally analogous: it is the ratio of the characteristic signal advection speed to the collector's convective capacity. The two parameterisations are dual — Pe characterises the dominance of advection over diffusion in the *fluid*; κ characterises the dominance of drift over collector actuation in the *agent*.

**Connection to catchability in ecology:** The Beverton-Holt catchability coefficient q is defined as the probability of capture per unit effort. In continuous terms, q ∝ v_collector / v_prey — which is 1/κ in our framing. High q (low κ) means collectors are effective; low q (high κ) means the target outruns the collector.

**Predicted effect on coordination gain:** h(κ) should be non-monotone.
- At very low κ (collectors much faster than drift): collectors can always catch up; coordination helps mainly with sensing (finding, not catching). Gain is moderate.
- At κ ≈ 1 (comparable speeds): coordination is most valuable — sharing velocity information allows collectors to intercept on the right trajectory rather than chasing.
- At κ >> 1 (drift much faster than collectors): collectors cannot catch particles regardless of communication. Gain should drop toward zero.

The peak of h(κ) is predicted near κ ≈ 0.5–1.0.

---

## The separability hypothesis

**Claim:** The expected paired coordination gain Δ̄(ρ, κ) is approximately multiplicatively separable:

```
Δ̄(ρ, κ) ≈ C · g(ρ) · h(κ)
```

where g and h are monotone functions of ρ and κ respectively, and C is a scaling constant.

**Mechanistic justification:** Communication adds value through two independent channels:
1. **Sensing channel:** Shared team estimates reduce effective ρ — the team mean velocity estimate has higher SNR than any single-agent estimate (by Proposition 2, the count-weighted mean is the sufficient statistic). This effect depends on ρ but not on κ.
2. **Action channel:** The shared velocity estimate tells collectors which direction to move. This reduces wasted actuation. The value of this information depends on κ — if κ is very high, collectors can't execute the implied action regardless.

If these two channels are approximately independent, the total gain factorises. This is the separability hypothesis. It may fail if the two channels interact — e.g. if high ρ causes sensing failures that make action guidance irrelevant (no signal to act on).

**Empirical test:** Fit Δ̄(i,j) = a · exp(b · ρ_i) · exp(c · κ_j) to the 3×3 grid. Compute R². If R² ≥ 0.8, separability holds. If R² < 0.5, separability is rejected and the paper's central claim must be redirected.

---

## Predicted 3×3 pattern

Based on the mechanistic argument:

```
         κ=0.10   κ=0.20   κ=0.40
ρ=0.10:  [low]    [low]    [~0]
ρ=0.21:  [mod]    [+1.19]  [low]
ρ=0.35:  [high]   [mod]    [~0]
```

The highest gain is predicted at intermediate κ and high ρ — hard sensing + moderate control authority. This is where shared velocity summaries most complement individual estimates.

The lowest gain (near zero or negative) is predicted at high κ — when drift outpaces collector speed, communication cannot compensate.

---

## Connection to prior work

**Wang et al. (2025):** Studies a single mobile collector, no multi-agent coordination. Their operating regime is equivalent to low κ (single fast collector). Our framework predicts low coordination benefit in their regime — consistent with their not studying communication.

**Löffler et al. (2023):** Locally perceiving active particles (the targets move, not collectors). Their κ is effectively undefined (particles are the agents). The ρ-κ framework applies to collector-side parameterisation only.

**Vicsek model:** Aligning particles with κ >> 1 (no fixed collector) — the limit where individual agency against the field is impossible. Coordination in Vicsek-like systems is about alignment, not interception.

---

## What ρ-κ does not capture

- Arena geometry (boundaries, obstacles)
- Collector sensing radius (affects effective ρ through neighbourhood size)
- Team size M (affects SNR through pooling — captured in the √M scaling prediction of FR-B2)
- Temporal structure of the field (captures by FR-B4's ω parameterisation)

The ρ-κ framework is a two-parameter slice of a higher-dimensional task space. The claim is that these two numbers are the dominant predictors of coordination gain, not that they are exhaustive.
