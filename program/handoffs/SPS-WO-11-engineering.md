# SPS-WO-11 Experiment Engineer handoff

**Date:** 2026-08-01

**Input snapshot:** `f62b073cf2a5b1e283aff2e27ccd6ec7fae55c3d`

**Scientific episodes:** zero.

## Implemented

- Added `configs/experiments/sps_c04_outcome_blind_design.yaml` with the frozen
  `T=1.34` endpoint, exact `67/134/268` mappings, all-row outcome rule,
  eligibility logic, and incidence thresholds.
- Upgraded the aggregation summary to schema v2 without modifying any immutable
  v1 artifact.
- Added exact post-history numerators, denominators, and rates.
- Encoded the 10-percentage-point arm-gap limit as absolute fraction `0.10`.
- Kept rescue and cancellation as uncapped treatment mechanisms.
- Excluded only affected collector-transition units from displacement mediation
  when collector reflection occurs.
- Added deterministic endpoint, eligibility, denominator, threshold, and
  all-row-yield tests.

## Verification

Task-scoped tests: `11 passed`. Full suite:

```text
PYTHONPATH=src:/tmp/sps-wo11-testdeps python -m pytest -q
179 passed, 5 skipped
```

The five skips are optional PyTorch tests. JSON schema parsing and
`git diff --check` passed. No scientific seed, eta sweep, or performance result
was produced.

## Next action

The engineering gate permits design of a separate preregistration work order.
It does not authorize the eta grid or any seed execution.
