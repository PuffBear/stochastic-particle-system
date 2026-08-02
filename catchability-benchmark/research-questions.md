# Research Questions: Catchability Benchmark

## Primary research question

What is the minimum two-axis parameterization (ρ, κ) needed to predict multi-agent coordination gain from task parameters alone — and does coordination gain collapse onto a separable surface g(ρ)·h(κ) across physically rescaled versions of the SPS task?

## Definitions

**ρ = α·√dt / σ** — per-observation signal-to-noise ratio of the latent field direction. Low ρ means a single agent's field estimate is noisy; shared team summaries have their highest value here.

**κ = α / v_max** — ratio of field drift speed to collector max speed. κ < 1: collector is faster than drift (catching is easy, finding is hard). κ > 1: drift outruns collector (coordination cannot compensate for inadequate speed).

**Confirmed anchor:** At SPS-C03 (α=0.06, v_max=0.30, dt=0.02, σ=0.06): ρ = 0.141, κ = 0.20. Δ̄ = +1.19 particles, 95% lower bound +0.459, N=32 seeds.

---

## Q1 — Collapse at fixed ρ (first experiment, 3 κ levels)

**Setup:** Three configurations at fixed ρ = 0.141 and κ ∈ {0.10, 0.20, 0.40}. Each configuration adjusts (α, v_max) jointly to hold ρ constant while varying κ. All structural parameters frozen at SPS-C03 values (N=256, M=4, dt=0.02, σ=0.06, steps=67).

| κ | v_max | α |
|---|---|---|
| 0.10 | 0.60 | 0.060 |
| 0.20 | 0.30 | 0.060 (C03) |
| 0.40 | 0.15 | 0.060 |

Note: at fixed ρ=0.141, α is fixed (α = ρ·σ/√dt = 0.060). Only v_max varies across κ levels.

**Hypothesis:** Coordination gain Δ̄ varies significantly across κ at fixed ρ. At κ=0.10 (fast collectors), agents can catch any particle they find — sensing is the bottleneck, not catching. At κ=0.40 (slow collectors), even accurate field knowledge cannot be acted upon. Gain peaks near κ=0.20.

**The collapse claim:** If physical rescalings that hold (ρ, κ) constant also hold Δ̄ constant, then (ρ, κ) is sufficient to parameterise coordination gain. This collapse is what makes the parameterisation *useful* — it transfers results across physically different systems.

**Estimand:** Δ̄(κ) = E[Y(shared_v2) − Y(independent)] at each κ level. 8 matched seeds per condition (seeds 8001–8008).

**Gate:** κ=0.20 must replicate C03 (Δ̄ within ±0.5 of +1.19). Κ=0.10 and κ=0.40 must differ from κ=0.20 by more than 0.5 particles for the κ axis to be informative.

---

## Q2 — Full (ρ, κ) grid and separability

**Setup:** 3×3 grid: ρ ∈ {0.10, 0.141, 0.25} × κ ∈ {0.10, 0.20, 0.40}. 8 seeds per cell (72 runs). See `experiments/grid-design.md` for exact (α, v_max) per cell.

**Hypothesis:** The surface is approximately multiplicatively separable:
```
Δ̄(ρ, κ) ≈ g(ρ) · h(κ)
```
where g is decreasing in ρ (harder sensing → more benefit from shared channel) and h is non-monotone in κ (peaks near κ=0.20, drops toward zero at high κ).

**Pre-registered test:** Fit log(Δ̄ + offset) = log(a) + b·log(ρ) + c·log(κ). Report R². Separability passes if R² ≥ 0.80.

**Why separability matters:** If the surface separates, practitioners can predict coordination gain from two numbers. If it doesn't, there is an interaction term — the value of shared sensing depends on whether you can act on it — which is also scientifically interesting but changes the paper claim.

---

## Q3 — Cross-domain transferability

**Setup:** If Q1 collapse holds — i.e. physical rescalings at fixed (ρ, κ) produce matching Δ̄ — then the (ρ, κ) grid cell predicts gain in a physically different domain parameterised at the same point.

**Demonstration:** Identify one real-world domain (from `theory/domain-mapping.md`) whose (ρ, κ) estimate falls within the grid. Simulate it at SPS structural parameters rescaled to match. Report whether the predicted Δ̄ from the grid matches the simulated result.

This is exploratory and cannot be pre-registered without committing to a specific domain. It is the "so what" section of the paper — showing the benchmark transfers.

**Candidate domain:** Agricultural UAV at moderate wind speed (ρ≈0.14, κ≈0.20) falls directly on the confirmed C03 cell. This is the clearest transfer candidate.

---

## Kill criteria

**Collapse kill (Q1):** Δ̄ does not vary significantly across κ at fixed ρ — the κ axis adds no information. If Δ̄ is flat across κ∈{0.10, 0.20, 0.40}, ρ alone predicts gain and the two-axis framing is unnecessary.

**Separability kill (Q2):** R² < 0.50 on the multiplicative fit after all 9 cells — gain cannot be factored and the clean parameterisation breaks down. Redirect: characterise the interaction term instead.

**Replication kill:** Cell (ρ=0.141, κ=0.20) deviates from C03 Δ̄ by more than ±0.5 particles — infrastructure problem; halt and diagnose before continuing.

**Coverage kill:** Fewer than 5/9 cells show positive Δ̄ — coordination is not reliably beneficial across the regime; the benchmark characterises a narrow phenomenon.
