# SPS-WO-04 Experiment Engineer Handoff

**Date:** 2026-07-31  
**Role:** Experiment Engineer  
**Result:** upstream oracle gate failed after its single bounded implementation repair; canonical first-interception task is marked for kill/redesign. No sharing, scientific timestep slice, power study, confirmation, MARL, or HPC job was run.

## Immutable evidence verification

The six SPS-P02 files were checked against `program/snapshots/2026-07-31-pre-repair.sha256`; all passed. Its negative interpretation is unchanged: 0/144 contact outcomes changed after exact-contact repair, every simultaneous lower bound remained negative, and passive/random controls descriptively matched or exceeded `local_flow_v1`.

## Engineering delivered

- `privileged_upstream_oracle` remains accepted for legacy evidence, while new work uses the accurate ID `true_field_upstream_control`.
- Added a distinct action-feasible full-state intercept solver. It observes current positions and known deterministic drift, solves bounded constant-velocity intercepts, assigns distinct targets, emits legal actions, and never receives future Brownian increments.
- Added per-collector-step diagnostics for visible/valid tracks, policy use, estimated/true velocity error, action alignment, wall proximity, path length, swept-area proxy, first-contact owner, absolute time, and censoring.
- Added exact coupled-Brownian aggregation utilities.
- Added a three-number permutation-invariant bounded team velocity summary and identical-shape independent/shared controllers. Unit tests cover bounds, masked-particle leakage, permutation invariance, action equivariance, and exact Brownian aggregation.
- Test command: `PYTHONPATH=src python3 -m unittest discover -s tests -v` — **85 tests passed**. `python3 -m compileall -q src analysis` passed.

## Dependency-gated runs

### SPS-P03 — failed implementation diagnostic

Command recorded in its manifest; runtime 34.67 seconds. The initial oracle used the last realized Brownian displacement as if predictive. No point passed. The run remains immutable but is not the decisive oracle gate.

Artifacts: `results/raw/SPS-P03-CORRECTION-GATES/`. Key checksums: episode summaries `590cbb...498e6`; diagnostics `6b1ef7...a93f0`; gate `eb4b3d...6793`.

### SPS-P04 — decisive corrected oracle gate

Preregistered config: `configs/experiments/sps_p04_oracle_repair.yaml`. The oracle input was repaired to current deterministic field drift, with no physical task change and no future noise. Runtime was approximately 34 seconds on Codex CPU.

No nonzero point passed:

| alpha / kappa | mean passive-adjusted gain | positive seeds | oracle steps earlier than stationary | oracle steps earlier than true-field control |
| --- | ---: | ---: | ---: | ---: |
| 0.03 / 0.25 | -0.00844 | 0/8 | 1.875 | 0.625 |
| 0.06 / 0.50 | -0.00563 | 1/8 | 3.000 | -0.500 |
| 0.12 / 1.00 | -0.01063 | 2/8 | 1.000 | -0.125 |

Artifacts: `results/raw/SPS-P04-ORACLE-STATE-REPAIR/`. Key checksums: episode summaries `809586...4ee`; diagnostics `32b474...160`; gate `f51281...b98`.

### Diagnosis

The canonical endpoint is first-event saturated. Across the diagnostic arms, first contacts commonly occurred in the first few steps; stationary averaged 5.78 steps across the grid and the corrected oracle 3.00. Every episode was uncensored, so the 400-step horizon never bound. Step diagnostics show no wall involvement at these early contacts. The oracle is strongly better under null, which makes the signal-minus-null, passive-adjusted gate structurally difficult: signal can improve passive transport more than it improves an already-fast oracle. This is an endpoint/initial-geometry design problem, not a compute shortage.

### SPS-P05 procedural deviation

A 64-particle task-change run was launched before the Program Director's stop message arrived. It exceeded the work order's one-repair allowance. Its raw files are preserved under `results/raw/SPS-P05-SATURATION-REPAIR/`, but `results/derived/SPS-P05-procedural-status.json` permanently excludes it from claims, manuscript decisions, design selection, and power calculations. This deviation does not alter the kill/redesign decision.

## Decision and next authorized action

SPS-C03 is **blocked** by SPS-P04. The next work order should preregister a redesigned endpoint/reset contract that gives actions measurable time to matter—without selecting among several post-outcome variants—and then run a fresh oracle gate. Sharing remains implemented but scientifically unexecuted. Coupled-noise utilities are tested, but timestep interpretation is deferred because the upstream task gate failed.

No HPC is needed. No prior raw result or review was modified.
