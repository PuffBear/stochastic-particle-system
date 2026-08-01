# SPS-WO-09 Research Lead handoff

**Date:** 2026-08-01  
**Decision:** select SPS-C04 for theory-first feasibility; no scientific seeds.

## Selected atomic question

For four decentralized collectors exchanging one three-scalar local-drift
message per control step, at what field-correlation-length to nominal-spacing
ratio does all-to-all averaging change from harming to helping unique team
captures by `T=1.34`, relative to independent local estimation?

## Primary estimand

For fresh matched seed `s`, `Delta_s(eta)=Y_s(global,eta)-Y_s(independent,eta)`.
The sole target is a grid-censored zero crossing of `E[Delta_s(eta)]`. A paper
requires supported adverse and beneficial regions in the predicted order; no
crossing means no supported phase boundary.

## Theory target

For one stationary-field component with normalized cross-agent correlations
`c_jk`, sum `S`, field variance `sigma_v^2`, and independent local-estimation
noise `tau^2`, derive and verify

`R_global-R_independent = sigma_v^2(1-S/M^2)-tau^2(1-1/M)`.

This is the estimator-side prediction. The control paper must then establish
whether the non-additive unique-capture objective shifts the yield crossover
because common actions increase redundant pursuit.

## Alternatives rejected

- Local/reliability-weighted fusion: too close to correlation-aware distributed
  monitoring and novelty-gated neighbor fusion; retain only as downstream
  mitigation after the failure mode is established.
- Intended-target broadcasting: classic allocation/deconfliction burden and not
  connected tightly enough to the valid WO-07 diagnostic.
- Learned communication graph: crowded AAMAS method space and no identified
  mechanism yet.
- Growing capture cascades: currently provisional physics and not clearly an
  AAMAS multi-agent paper.

## Continuation rule

Proceed only through the SPS-WO-09 theory, field-covariance, communication, and
deterministic mechanism gates. Freeze no `eta` grid, effect threshold, seed
count, or inference rule using WO-07 outcomes. Learned policies, confirmation,
and HPC remain blocked.

