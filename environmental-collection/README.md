# Environmental Collection Planning Decision Layer

**Industry idea:** FI-2
**Status:** Unvalidated — domain partner and historical dataset required
**Readiness:** Low

## Problem

Environmental engineers deploying mobile collection systems face an unanswered planning question: will moving the collector actually collect more than letting the current carry material to a stationary device? Existing hydrodynamic models (EFDC, WASP, Delft3D) simulate transport — they don't model the decision value of collector mobility.

The gap: "Should we pay for autonomous mobility?" is a capital procurement question answered today by intuition or coarse analogy to past programs.

## The κ parameter as the key insight

The SPS research defined catchability: **κ = α / v_max** (effective field transport speed / collector max speed).

- **κ < 1:** collector cannot outrun field-advected material → mobility provides limited additional benefit
- **κ > 1:** collector can close on material → mobility provides substantial benefit
- **κ ≈ 1:** the boundary — marginal benefit of mobility is highest here

For environmental contexts: α is the effective transport speed of the target substance; v_max is the autonomous system's top speed. A plume moving at 0.3 m/s with a drone capable of 0.4 m/s gives κ ≈ 0.75 — mobility helps, but less than the drone's speed advantage implies because the task is collection, not interception.

## Files

- `business-questions.md` — customer segment, decision context, validation plan, revenue model, constraints
