# Research Questions: Catchability Benchmark

## Primary research question

At fixed sensing difficulty ρ = α√dt/σ, do team coordination gain curves collapse across physically rescaled versions of the SPS task when indexed by control authority κ = α/v_max? If so, what is the minimum two-axis parameterization (ρ, κ) needed to predict coordination gain from task parameters alone?

## Definitions

**ρ = α · √dt / σ** — how clearly can an agent detect the field direction from local particle velocities?

**κ = α / v_max** — can the collector outrun a particle moving at field velocity? κ < 1: collector is slower; κ > 1: collector can close on any particle.

## Sub-questions

### Q1 — Collapse at fixed ρ

**Setup:** Three rescaled configurations at fixed ρ = 0.21 and κ ∈ {0.10, 0.20, 0.40}. Rescaling adjusts (α, v_max, σ) simultaneously while holding ρ constant. All structural parameters (N, M, K, arena, evaluation_steps) fixed.

**Hypothesis:** Coordination gain Δ̄ varies significantly across κ at fixed ρ. At κ=0.10, even accurate field knowledge produces little gain because the collector cannot close on particles. At κ=0.40, gain is higher because the collector can exploit field direction more aggressively. The collapse hypothesis: gain curves from different physical instantiations lie on the same κ-indexed curve at matched ρ.

**Estimand:** Δ̄(κ) = E[Y(shared) − Y(independent)] at each κ level. 8 seeds per condition.

### Q2 — Predicting gain from (ρ, κ)

**Setup:** A 3×3 grid: ρ ∈ {0.10, 0.21, 0.40} × κ ∈ {0.10, 0.20, 0.40}. 8 seeds per cell (72 total).

**Hypothesis:** The surface is approximately separable: Δ̄(ρ, κ) ≈ g(ρ) · h(κ), where g is increasing in ρ and h is an inverted-U in κ. Fit a product model and report R².

### Q3 — Cross-domain transferability

If Q1 collapse holds, demonstrate that SPS results at (ρ=0.21, κ=0.20) predict coordination gain in a physically different domain parameterized at the same point.

## Kill criteria

- **Collapse kill:** Gain does not vary significantly across κ at fixed ρ — ρ alone determines coordination gain; the two-axis framing adds no value.
- **Separability kill:** No separable structure in the (ρ, κ) surface — coordination gain cannot be factored and the parameterization provides no simplification.
