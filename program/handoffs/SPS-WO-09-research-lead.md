# SPS-WO-09 Research Lead handoff

**Date:** 2026-08-01  
**Decision:** conditional analytic possibility gate passed; SPS-C04 remains
unsupported and no scientific seeds are authorized.

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

## General estimator result

Let `B=Cov(V)` be the covariance of latent local summaries and
`Omega=Cov(E)` the covariance of zero-mean summary errors, independent of `V`.
The all-to-all minus self-only average estimator risk is

`D = tr(B)/M - 1'B1/M^2 + 1'Omega1/M^2 - tr(Omega)/M`.

The former independent homoskedastic formula is a special case. Correlated
errors reduce or remove pooling's denoising benefit.

For the actual particle-average sensor, conditional on positions and valid sets,

`B_ij=(n_i n_j)^-1 sum_{p in S_i,q in S_j} C_ell(X_p-X_q)`

and, away from clipping/reflection/fallback,

`Omega_ij=(sigma_D^2/dt) |S_i intersection S_j|/(n_i n_j)`.

These finite-sensing and overlap terms are now implemented and tested.

## Conditional crossover check

For equal marginal field variance and independent equal-variance errors, a
unique point-sensor squared-exponential crossover exists only when
`0 < tau^2/sigma_v^2 < 1`. With four fixed square centers and ratio `0.5`, the
illustrative threshold is `eta*=0.95980`. At `eta=0.5`, `D=+0.302753 sigma_v^2`;
at `eta=2`, `D=-0.260949 sigma_v^2`. An error correlation of `0.4` shifts the
illustrative threshold to `1.35199`; changing square geometry to a line with
the same nominal spacing shifts it to `0.71261`. Therefore `eta` alone is not
a universal causal variable: the benchmark claim must condition on its frozen
geometry distribution and actual sensed-summary covariances.

This establishes mechanism possibility only. The control paper must still show
whether the non-additive unique-capture objective changes sign because common
actions alter redundant pursuit. The general occupancy identity is
`E[U]=sum_p P(union_i {i captures p})`; the two-agent
`2-P(same target)` expression is only a special case.

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

## Gate decision and continuation rule

The theory gate conditionally passes. The exact finite-basis field, aggregation
primitive, covariance calculation, and deterministic microcases also pass.
However, runtime field integration, matched-stream logging, and nonlinear
clipping/reflection/missingness audits are still absent. Freeze no `eta` grid,
effect threshold, seed count, or inference rule until those gates close. WO-07
outcomes remain barred from selecting them. Learned policies, scientific
episodes, confirmation, and HPC remain blocked.
