# SPS-WO-02 Engineering Handoff

**Date:** 2026-07-31  
**Status:** bounded environment slice passed; overall correctness pilot remains open  
**Scientific claim status:** unchanged; all paper claims remain proposed

## Implemented

- bounded holonomic collector dynamics with Euclidean action clipping and exact reflection;
- deterministic collector lattice, with canonical four-agent order
  `(0.25,0.25),(0.25,0.75),(0.75,0.25),(0.75,0.75)`;
- seeded rejection sampling for 256 particles with strict no-capture reset;
- causal local observations with `K=32` nearest free particles, consecutive-visibility velocity masks, normalized self/relative positions, and no field/global/future-noise inputs;
- minimal reset/step/capture/reward/termination/truncation environment;
- one-based first-contact logging;
- fifth independent random stream for policy actions;
- stationary, complete pre-generated paired-random, and privileged upstream-field smoke policies;
- trajectory-step and reproducibility-manifest schemas.

## Verification

```text
python3 --version
python3 -c 'import numpy; print("NumPy", numpy.__version__)'
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
python3 .../research_program.py check .
```

Verified environment: Python 3.12.13; NumPy 2.3.5. Final result: 44 tests passed, zero failures or errors; byte-compilation passed; program-structure validation passed.

Coverage includes 100 canonical reset seeds, exact reset reproducibility, action/speed/reflection limits, causal newly-visible velocities, field/future-noise/out-of-radius leakage, fixed teammate ordering, one-draw paired field orientation, paired random-action identity, zero-signal complete rollout identity, zero-diffusion stationary limit, capture ownership, one-based first contact, horizon truncation, schema parsing, and a deterministic high-signal oracle calibration microcase.

## Retained failures and corrections

The first provisional implementation passed 36 tests but failed root contract review: collectors were initialized randomly, newly visible particles received noncausal velocity, teammate information used the wrong contract, and the privileged diagnostic followed rather than opposed the uniform field. The implementation and tests were corrected. A later contract-edit run intentionally exposed one stale assertion and three stale-test errors; those tests were corrected to the frozen contract without rolling back production behavior.

## Interpretation

This is engineering evidence only. The smoke policies establish that the environment can execute bounded limiting cases. They do not estimate a detectability boundary, compare policy performance, establish coordination, validate aggregation, or support AAMAS acceptance.

## Remaining blockers

1. Replace stateful tie RNG consumption with event-keyed tie breaking for causally coherent divergent matched episodes.
2. Resolve endpoint-only contact with continuous segment handling, including reflected subpaths, or freeze a timestep-convergence criterion.
3. Implement a matched null/signal episode wrapper that validates complete provenance.
4. Freeze and test `local_flow_v1` using only causal masked observations.
5. Implement trajectory writing, hashes, and manifest validation.
6. Freeze growing attached-disc radius, motion, wall, and visualization semantics before any aggregation analysis.
7. Add a standard PettingZoo/Gym adapter only after the primary core remains stable.
