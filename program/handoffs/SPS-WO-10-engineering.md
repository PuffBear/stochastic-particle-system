# SPS-WO-10 Experiment Engineer handoff

**Date:** 2026-08-01

**Input snapshot:** `d4a497ab50021d2ed17289d1ba56cf420075947a`

**Result:** deterministic integration gate passed; SPS-C04 remains unsupported.

## Delivered

- Strict YAML/reset construction of an episode-frozen periodic Gaussian field
  from the dedicated field stream, including field seed and realization hash.
- Stable top-`K` diagnostic particle IDs without adding IDs or latent state to
  `LocalObservation`.
- Retained transition-source positions, latent velocities, and reflection flags
  aligned with each apparent-velocity message.
- One exact policy trace containing raw and clipped messages, aggregation,
  empty/fallback/rescue/cancellation state, and the actual actions.
- A separate aggregation runner for self-only versus all-to-all under the same
  nonzero field; legacy null-versus-signal semantics were not repurposed.
- Matched hashes for initialization, Brownian tensor, field realization, policy
  randomness, and event-key provenance.
- Schema-valid source/outcome geometry, valid-set overlap, covariance, action,
  event, nonlinear-incidence, and cumulative-yield diagnostics.
- Conditional finite-sensing `B`, overlap-error `Omega`, and risk `D`, with
  `signal_strength^2` scaling and explicit null/reasons for ineligible rows.
- Pre-simulation collision checks and numeric JSON-schema bound validation.

## Verification

- Frozen SPS-WO-10 command: **74 tests passed**.
- Full available suite: **171 passed, 5 optional PyTorch skips**.
- JSON parsing, Python compilation, `git diff --check`, and research-program
  integrity check passed.
- The worker's integration fixtures used four arm-microepisodes and eight
  transitions. Program-Director regression validation retained no result
  artifact and generated no scientific evidence.
- Scientific episodes: **0**.
- No file under `results/`, claim ledger, experiment ledger, manuscript, grid,
  endpoint, effect threshold, seed budget, or inference rule changed.

## Corrected implementation issue

An initial communication regression exposed that replacing the explicit 4x4
adjacency bounds with only a schema reference broke the direct contract audit.
The inline bounds were restored; all seven communication tests pass.

## Boundary and next action

Engineering **GO** means only that outcome-blind design work may follow. It does
not authorize a pilot. Three blockers remain:

1. close the primary-literature veto;
2. resolve `T=1.34` versus canonical `400*0.02=8.0` prospectively, including
   timestep endpoint matching; and
3. freeze a handling rule and incidence/error budget for analytically
   ineligible clipping, particle-reflection, empty-summary, and fallback rows.

No HPC access is needed.
