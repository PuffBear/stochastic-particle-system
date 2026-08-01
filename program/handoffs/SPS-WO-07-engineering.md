# SPS-WO-07 Experiment Engineering Handoff

**Date:** 2026-08-01  
**Role:** Experiment Engineer  
**Base commit:** `44fa041e866781744ed10f50864ea529f0c95990`  
**Status:** valid negative diagnostic; joint gate failed

## Pre-outcome repair and contract

Before any WO-07 seed was opened, an independent audit found that the original
passive-adjusted gate was algebraically identical to the primary contrast and
that the two-particle minimum relevant effect was not enforced. The runner,
config, work order, registry, and deterministic tests were repaired and
published at the base commit above. The repaired GitHub CI passed before
execution. The output path remained absent throughout repair.

The repaired six-component joint gate required:

1. correctness and complete execution;
2. matched stochastic streams;
3. mean shared-minus-independent yield at least 2.0 particles;
4. positive shared-minus-independent yield in at least 5/8 seeds;
5. mean shared-minus-stationary yield above zero;
6. mean signal-specific shared-independent difference-in-differences above zero.

All components were frozen before seeds 4001--4008 were consumed.

## Exact execution

```text
PYTHONPATH=src:. python -m unittest tests.test_experiment_design -v
PYTHONPATH=src:. python analysis/run_attribution_controls.py --output results/raw/SPS-WO-07-ATTRIBUTION-CONTROLS --config configs/experiments/sps_wo07_attribution_controls.yaml --repository-base-commit 44fa041e866781744ed10f50864ea529f0c95990 --upstream-gate results/raw/SPS-WO-06-TIMESTEP-CONVERGENCE/convergence_report.json
PYTHONPATH=src:. python -m unittest tests.test_experiment_design -v
```

The run executed exactly once, completed 64 episodes and 4,288 environment
steps in 251.445 seconds on CPU, and emitted 743 capture events. All 11 design
tests passed before and after execution.

## Result

The primary shared-minus-independent seed effects were:

```text
[3, 6, 4, -3, 0, 0, 5, -1]
```

- mean: 1.75 particles;
- median: 1.5;
- sample standard deviation: 3.196;
- descriptive paired-bootstrap 95% interval: `[-0.375, 3.75]`;
- positive direction: 4/8 seeds.

The mean and direction gates failed. The other four components passed:

- correctness and execution: true;
- matched streams: true;
- shared-minus-stationary mean: 1.875;
- signal difference-in-differences mean: 2.125.

Oracle-minus-shared mean was 7.5 and positive in 8/8 seeds. These secondary
observations cannot rescue a failed joint gate.

## Immutable artifacts

`results/raw/SPS-WO-07-ATTRIBUTION-CONTROLS/` contains:

- `episode_summaries.jsonl`: 64 rows; SHA-256 `2630b82389f971d73ef2db88e8b01606fabdb6c88d79fe4193fd9b090f905225`;
- `capture_events.jsonl`: 743 rows; SHA-256 `041ca9cf5292a315849b3fc7fa3f928d47d6e2661bd5630feca330ef8ab77e83`;
- `attribution_gate.json`: SHA-256 `9111a3b204a3f31ea53ad87fc4c14370fb7502fde7666cd39beb4861fbd24c83`;
- `manifest.json`: SHA-256 `5fd6e7907adbb2231c773fb020b71509a9752bd5b6b3890fb2764ba74d079c6a`.

Independent verification matched every artifact byte count and hash, all 64
condition keys, summed yield/event accounting, source hashes, matched streams,
and the recomputed six-component gate.

## Interpretation and stop point

This is a valid negative diagnostic. The tested three-number shared statistic
did not show the frozen practically relevant and seed-consistent advantage. It
is not a statistical refutation of every sharing mechanism, because the sample
was diagnostic and its interval includes zero. SPS-C03 is dropped rather than
confirmed or statistically refuted.

WO-08, confirmation, additional attribution seeds, and MARL are not authorized.
Any future mechanism requires a new theory-driven claim, work order, and fresh
seed block.
