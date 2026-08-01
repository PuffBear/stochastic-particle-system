# SPS-WO-01 Engineering Handoff

**Date:** 2026-07-31  
**Snapshot:** local task-scoped implementation pending GitHub publication  
**Claim status:** unchanged; all scientific claims remain proposed

## Objective

Implement the smallest correctness-first slice needed before a simulator
performance pilot: independent random streams, exact reflecting boundaries,
null/uniform/vortex fields, matched free-particle propagation, permanent
fixed/growing capture primitives, and one-based first-interception metrics.

## Created artifacts

- `pyproject.toml`
- `configs/env/canonical.yaml`
- `configs/experiments/pilot.yaml`
- `configs/experiments/detectability_protocol.yaml`
- `src/particle_benchmark/seeding.py`
- `src/particle_benchmark/dynamics/{boundaries,fields,particles,capture}.py`
- `src/particle_benchmark/metrics/episode.py`
- `tests/test_core.py`

## Verification

```text
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Result: 17 tests passed under Python 3.12.13 and NumPy 2.3.5. The project
contract remains Python 3.11 compatible but has not yet been executed in a
Python 3.11 environment.

## Failure retained

The first growing-capture microcase initially failed because the second test
particle was outside the attached node's capture radius. The test fixture was
corrected from x=0.16 to x=0.14; the implementation was not changed to force a
pass. The corrected microcase verifies next-step capture through the attached
node.

## Interpretation

These are code-integrity observations only. They establish no detectability,
coordination, aggregation, dataset, or MARL result.

## Next action

Implement collector motion, deterministic initial-state sampling, local
observations with leakage tests, and an end-to-end micro-environment before any
scripted performance pilot.

