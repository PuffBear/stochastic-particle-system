# SPS-WO-06 Experiment Engineering Handoff

**Date:** 2026-08-01  
**Role:** Experiment Engineer  
**Base commit:** `2742ef387ad70fdc2b9cc84558bfd9b87e54b3a7`  
**Status:** valid diagnostic gate passed; SPS-WO-07 authorized but not run

## Contract and commands

The immutable output path was absent before execution. The frozen configuration,
seeds, timestep levels, physical duration, policies, primary tolerance, sign
rule, and compute cap were not changed.

```text
PYTHONPATH=src:. python -m unittest tests.test_experiment_design -v
PYTHONPATH=src:. python analysis/run_timestep_convergence.py --output results/raw/SPS-WO-06-TIMESTEP-CONVERGENCE --config configs/experiments/sps_wo06_timestep_convergence.yaml --repository-base-commit 2742ef387ad70fdc2b9cc84558bfd9b87e54b3a7
PYTHONPATH=src:. python -m unittest tests.test_experiment_design -v
```

All seven design tests passed before and after the run. An optional attempt to
use `/usr/bin/time -v` failed because that binary was unavailable; Python did
not start, no seed was consumed, and no artifact path was created during that
attempt. The exact registered command then ran once.

## Execution and result

The CPU run completed 96 episodes and exactly 15,008 environment steps in
853.541 seconds on Python 3.12.13 and NumPy 2.3.5. It used diagnostic seeds
3001--3008, `alpha` in `{0, 0.06}`, stationary and full-state-oracle policies,
and coupled Brownian paths at `dt` 0.02, 0.01, and 0.005 over physical duration
1.34. These seeds and outcomes are permanently ineligible for confirmation.

Oracle-minus-stationary means at `alpha=0.06` were:

- `dt=0.02`: 8.625 particles, 8/8 positive seeds;
- `dt=0.01`: 8.625 particles, 8/8 positive seeds;
- `dt=0.005`: 8.375 particles, 8/8 positive seeds.

The primary absolute mean difference was 0.0 particles, strictly below the
frozen one-particle tolerance, and 0/8 seed directions changed. The mandatory
finest-level mean differed from `dt=0.01` by 0.25 particles and was
informational. All correctness, coupling, provenance, and artifact gates passed.

## Immutable artifacts

`results/raw/SPS-WO-06-TIMESTEP-CONVERGENCE/` contains:

- `episode_summaries.jsonl`: 96 rows; SHA-256 `f6f3dbb6bb8812e96d43e965933e2e3c7bcab09638af1e24ea602762e27592ba`;
- `capture_events.jsonl`: 1,357 rows; SHA-256 `55a1c1df95d6b5793279e00636be96181ae5bae730cf759f2683f116fe400b83`;
- `convergence_report.json`: SHA-256 `5051950452be6c3a3fc0640084234d068d0dd0e49305a99bfe7816e108947158`;
- `manifest.json`: SHA-256 `3648181a1015edef2aff423b8a7569d7e4d8c7cf2f1cbff083b0ecaee0d108d7`.

An independent audit checked every manifest hash and byte count, all 96 unique
condition tuples, event accounting, exact coupled-increment reconstruction,
the frozen report recomputation, and all gate components.

## Scientific interpretation and next action

The result is high-confidence diagnostic evidence that the previously observed
oracle-versus-stationary action headroom is not an obvious artifact of the
canonical timestep. It is not evidence that sharing, communication,
coordination, or MARL helps. SPS-C03 remains blocked.

The only authorized next scientific run is SPS-WO-07, after one final frozen
config, seed-firewall, policy-ID, and matched-stream audit. Confirmation, power
selection using diagnostic outcomes, and MARL training remain blocked.
