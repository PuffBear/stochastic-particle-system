# Catchability Benchmark: A Nondimensional Two-Axis Parameterization

**Research idea:** FR-B3
**Target venue:** ICML 2027 (benchmark track)
**Status:** 2–3 months out; requires SPS fixed-geometry results published

## Core contribution

The SPS benchmark conflates two independent axes of task difficulty:

- **ρ = α√dt/σ** — sensing difficulty (signal-to-noise ratio of the latent field)
- **κ = α/v_max** — control authority (how catchable field-advected targets are)

Changing α moves both axes simultaneously. A result at α=0.06 is not directly comparable to a result at α=0.12 without holding ρ and κ separately.

This paper proposes the two-axis (ρ, κ) parameterization, shows how to hold one axis fixed while varying the other, and validates that coordination gain curves collapse across physical rescalings at fixed ρ when indexed by κ. If they collapse, SPS results become transferable across physical domains. If they don't, the paper identifies what additional factor is missing.

**At SPS-C03:** ρ ≈ 0.21, κ ≈ 0.20 — both in the low-authority, low-SNR regime.

## Physical domain examples

| Domain | ρ analog | κ analog |
|---|---|---|
| Water-quality sampling drones | SNR of tracer concentration gradient | Drone speed / river current speed |
| Aerial wildlife surveillance | Detection SNR of animal movement | UAV speed / animal speed |
| Warehouse robot collection | Sensor noise on item flow direction | Robot speed / conveyor speed |

## Files

- `research-questions.md` — collapse hypothesis, sub-questions, kill criteria
