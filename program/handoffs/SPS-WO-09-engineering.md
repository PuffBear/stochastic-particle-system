# SPS-WO-09 Experiment Engineer handoff

**Date:** 2026-08-01
**Scope:** deterministic theory support and software only; zero scientific episodes

## Delivered

- `EpisodeFrozenGaussianField`: immutable finite Fourier Gaussian vector field
  on the arena torus with declared correlation length, exact evaluable
  covariance, invariant component marginal variance, and canonical SHA-256.
- Pure three-scalar message validation and `independent`/`all_to_all`
  aggregation; the policy encoder and decoder are identical across arms.
- General estimator-risk primitive plus exact conditional sensing-kernel and
  overlapping-particle error covariance construction.
- Strict four-agent message diagnostic schema with stream and field provenance.
- Constant-message equivalence, opposing-region cancellation, independent-
  noise denoising, correlated-error reversal, sensing-overlap,
  permutation-equivariance, bounds, immutability, and fingerprint tests.

## Verification

- Frozen work-order command: **29 tests passed**.
- Full available suite: **162 passed, 5 skipped**; skips require optional
  PyTorch and do not cover WO-09 primitives.
- `git diff --check`: clean.
- Deterministic field realizations remained below the work-order cap of 64.

These are engineering and analytic checks, not scientific outcomes.

## Blocking boundary

The field can be evaluated with
`field_velocity(..., family="periodic_gaussian", frozen_field=...)`, but the
standard config/environment/reset/runner pipeline cannot yet instantiate and
pair it. Those files were outside WO-09's allowlist. Runtime diagnostic writing
is likewise absent. Consequently, no pilot or performance seed is authorized.

## Required successor

Authorize a separate deterministic integration work order covering config,
environment reset, matched runner, schema writer/validator, and tests. It must
prove byte-identical initial state, Brownian tensor, field realization, and tie
stream across arms; quantify clipping, reflection, empty-valid-set fallback,
and moving-geometry incidence; then expose actual conditional `B` and `Omega`.
Only after that audit may the Program Director freeze a grid or seed budget.
