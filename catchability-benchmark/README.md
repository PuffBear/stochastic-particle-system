# Catchability Benchmark: A Nondimensional Two-Axis Parameterization

**Research idea:** FR-B3
**Target venue:** ICML 2027 (primary) / NeurIPS 2027 D&B (fallback)
**Status:** Ready to run — zero new code required; parameter sweeps over confirmed SPS infrastructure

## Core contribution

The SPS-C03 confirmation established a positive coordination effect at one operating point. But α=0.06 conflates two independent axes of task difficulty:

- **ρ = α·√dt / σ** — sensing difficulty (per-observation SNR of the latent field direction)
- **κ = α / v_max** — control authority (how catchable field-advected targets are)

Changing α alone moves both axes simultaneously. A result at α=0.06 is not directly comparable to one at α=0.12 without separately holding ρ and κ.

This paper proposes the (ρ, κ) parameterization, tests whether coordination gain curves collapse across physical rescalings at fixed ρ when indexed by κ, and validates the separability hypothesis Δ̄(ρ, κ) ≈ g(ρ)·h(κ) across a 3×3 grid. If the surface is separable, SPS results become transferable across physical domains. If not, the paper identifies what additional structure is missing.

**Note on ρ at SPS-C03:** At α=0.06, dt=0.02, σ=0.06, the formula gives ρ = α·√dt/σ = √0.02 ≈ 0.141. Earlier project documents cited ρ≈0.21 — this appears to be an arithmetic error in those documents. All experiments and analysis use ρ = 0.141 as the confirmed anchor.

## Physical domain examples

| Domain | ρ analog | κ analog |
|---|---|---|
| Agricultural UAV (pest spores) | Wind direction SNR from particle density | UAV speed / wind speed |
| Water-quality AUV sampling | Tracer concentration gradient SNR | AUV speed / current speed |
| Warehouse robot collection | Sensor noise on item flow direction | Robot speed / conveyor speed |
| Aerial wildlife surveillance | Detection SNR of animal movement | UAV speed / animal speed |

## Files

- `research-questions.md` — collapse hypothesis, 3×3 grid sub-question, transfer sub-question, kill criteria
- `PLAN.md` — seven-phase publication plan to ICML 2027 submission
- `theory/parameterization.md` — first-principles derivation of ρ and κ, separability hypothesis
- `theory/domain-mapping.md` — five real-world domains mapped to (ρ, κ) space
- `experiments/grid-design.md` — pre-registered 3×3 grid, seed plan, analysis protocol
- `paper/outline.md` — full 8-page ICML section outline with draft abstract
