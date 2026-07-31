# SPS-WO-05 Experiment Engineering Handoff

**Date:** 2026-07-31  
**Role:** Experiment Engineer  
**Scope:** fixed-horizon unique team capture-yield endpoint and the frozen 48-episode diagnostic only  
**Status:** valid R1 gate passed; downstream diagnostic authorized but not started

## Contract respected

I implemented the inclusive step-67 unique-team-capture endpoint as a bounded analysis runner over the unchanged environment and unchanged policies. Episodes no longer stop after first contact. Reset, particle count, collector count, physics, capture geometry, field, action rules, and policy parameters were not changed.

SPS-P05 outcomes were not opened, summarized, pooled, cited, or used. Only its previously recorded permanent procedural exclusion was respected.

## Implementation and tests

Created:

- `analysis/run_yield_gate.py`;
- `configs/experiments/sps_wo05_yield_gate.yaml`;
- `tests/test_yield_gate.py`.

The new deterministic tests cover unique counting, inclusion of step 67, exclusion of step 68, continued execution after first contact, and exact matched null/signal stream provenance. The full suite passed **89 tests** and `python -m compileall -q src analysis tests` passed. No reset, physics, or policy source file was modified.

Verification commands:

```text
PYTHONPATH=src:. python -m unittest tests.test_yield_gate -v
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src analysis tests
```

## Execution-control race and canonical evidence

The first tool session returned before its process reached a terminal state and was prematurely recorded as invalid. The process continued in the background and wrote a complete bundle 54 seconds later, after the identical R1 process had already been launched. The two runs have bitwise-identical episode summaries and capture-event files. R1 is the sole canonical evidence package; the late-completing original is excluded as a duplicate and is not pooled, counted as replication, or used to improve precision. The immutable status record is retained, and the correction is documented in `results/derived/SPS-WO-05-execution-race-audit.json`.

## Valid frozen execution

Exact command:

```text
PYTHONPATH=src:. python analysis/run_yield_gate.py --output results/raw/SPS-WO-05-YIELD-GATE-R1 --config configs/experiments/sps_wo05_yield_gate.yaml --repository-base-commit 1289499b4d00c7384c8364cc534f69ba552d87df
```

The run completed in 190.846 seconds on Python 3.12.13 / NumPy 2.3.5. It executed exactly 48 episodes and 3,216 environment steps on seeds 2001--2008, alpha in `{0,0.06}`, and the three frozen policies. All 48 episodes continued past their first contact to step 67. No GPU or HPC was used.

## Gate result

Primary oracle-minus-stationary seed contrasts at alpha 0.06 were:

```text
[11, 13, 9, 12, 7, 7, 11, 5]
```

- mean: **9.375** unique particles;
- median: **10**;
- sample standard deviation: **2.825**;
- descriptive paired-bootstrap 95% interval: **[7.5, 11.25]**;
- positive direction: **8/8 seeds**.

Oracle-minus-true-field-control contrasts were:

```text
[6, 12, 6, 15, 8, 3, 11, -1]
```

- mean: **7.5**;
- median: **7**;
- sample standard deviation: **5.155**;
- descriptive paired-bootstrap 95% interval: **[4.125, 10.875]**;
- positive direction: **7/8 seeds**.

Every frozen component passed: correctness/execution, mean gain at least four, positive direction in at least six seeds, positive targeting mean, matched streams, checksums, and artifact completeness. The 8-seed intervals are descriptive, not confirmatory.

## Immutable artifacts and provenance

`results/raw/SPS-WO-05-YIELD-GATE-R1/` contains:

- `episode_summaries.jsonl`: 48 rows; SHA-256 `1ae5a80b06b783ad11b044b48ed37e82fda21e7f0eb52f09daaabe4621349b3c`;
- `capture_events.jsonl`: 513 events with step, particle, owner, and contact fraction; SHA-256 `6621bc43d4fbd14c7f1f07cacfb5521db999afc90a2fa34d4ab1a966c4e72724`;
- `oracle_gate.json`: immutable gate summary; SHA-256 `a54d75875b8fbca4a9b1b50e4336389a8de0a12f0cbc88b4fa87ff6ff9581226`;
- `manifest.json`: environment, frozen config, source checksums, base revision, runtime, and artifact checksums;
- `execution_provenance_supplement.json`: exact shell invocation including `PYTHONPATH` and Python executable.

Every episode records unique yield, per-collector counts, every capture event, per-collector and total path length, matched initialization/Brownian/field/tie checksums, and confirmation-ineligible status. All manifest artifact hashes were independently recomputed and matched.

## Scientific interpretation and stop point

The positive gate means the unchanged canonical task has practically relevant action-contingent headroom under the revised fixed-horizon endpoint. It authorizes a separately preregistered coupled-noise timestep validation and only then a bounded sharing diagnostic.

It does **not** establish that communication or coordination helps, does not support SPS-C03, and cannot enter confirmatory estimation. Seeds 2001--2008 and all WO-05 outcomes remain permanently diagnostic. Per instruction, no sharing, timestep, power, learned-policy, MARL, or confirmatory run was started.
