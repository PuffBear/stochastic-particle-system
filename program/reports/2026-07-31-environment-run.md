# Stochastic Particle Lab — Immediate Environment Run

**Date:** 31 July 2026  
**Active paper:** one  
**Stage:** end-to-end environment core passed with unresolved correctness blockers  
**Scientific result status:** none  
**Repository branch:** `research-autonomy`

## TL;DR

1. The requested “tomorrow” environment cycle was completed immediately: bounded collector movement, deterministic capture-free resets, causal local observations, a minimal end-to-end environment, three smoke policies, and dataset schemas now exist.
2. The final validation suite has 44 passing tests, including 100 capture-free reset seeds, zero-signal full-rollout identity, causal velocity visibility, leakage checks, paired random actions, field-orientation matching, and limiting cases.
3. I rejected an earlier 36-test implementation despite its green test suite because it violated the frozen contract in scientifically dangerous ways. The corrected version now uses the exact deterministic lattice, causal newly-visible velocities, relative teammate positions, and an upstream privileged oracle.
4. The updated four-page manuscript compiles and still makes no empirical claim. All three scientific claims remain `proposed`; no dataset, performance sweep, detectability estimate, or coordination result was generated.
5. The fresh AAMAS review remains **2/10 Reject, confidence 4/5**. Its most important new finding is that endpoint-only capture can miss within-step crossings, so continuous-contact handling or timestep-convergence evidence is now a mandatory blocker before performance pilots.

## Explain it simply

The simulator can now run a complete episode rather than only moving free particles in isolation. Four collectors start in fixed, reproducible positions. The particles start randomly, but the reset procedure guarantees that none is already touching a collector. Every collector receives only a local view: nearby free particles, apparent particle velocity only when that particle was visible in two consecutive observations, and the relative positions of teammates.

The same scenario seed creates the same particle placement, Brownian motion, field orientation, tie stream, and pre-generated random-policy actions. At zero signal, a null-field episode and a uniform-field episode now remain identical across a complete rollout under the paired action tensor.

That means the environment is substantially closer to being a trustworthy measuring device. It does **not** mean that a weak flow is detectable. We have not yet implemented the research policy or estimated any boundary.

## What changed

### Collector dynamics

For collector action `a`, the action is projected into the unit Euclidean ball and converted into physical velocity using `v_max=0.12`. With `dt=0.02`, commanded displacement before reflection is at most `0.0024` per step. Non-finite actions fail closed. Collectors use the existing exact reflected-boundary primitive.

### Deterministic reset

The canonical collector IDs are frozen at:

1. `(0.25, 0.25)`;
2. `(0.25, 0.75)`;
3. `(0.75, 0.25)`;
4. `(0.75, 0.75)`.

Particle locations are drawn only from the initialization stream and rejected when they fall within the capture radius plus numerical clearance. A dedicated test checks all 256 particles across 100 deterministic reset seeds.

### Causal local observation

Each collector receives:

- normalized self position;
- teammate-relative positions normalized by the arena and ordered by collector ID;
- up to `K=32` nearest visible free particles, sorted by distance and hidden internal ID;
- for each particle: relative x/y position divided by sensing radius, apparent endpoint x/y velocity, and normalized radial distance;
- particle-presence and velocity-validity masks.

Velocity is emitted only if the particle is visible to that collector in both the previous and current observation. A newly visible particle receives zero velocity and an invalid-velocity mask. The observation API has no field, signal strength, field orientation, global state, random generator, future-noise, or privileged-particle input.

### Environment step order

1. Validate and move collectors.
2. Move every free particle using the current pre-generated Brownian slice and current field.
3. Resolve capture from post-motion positions using the pre-step aggregate geometry.
4. Permanently assign ownership and log first contact as one-based step `t+1`.
5. Recompute causal visibility and construct the next observation.
6. Truncate at the frozen horizon; first contact does not normally terminate the episode.

### Smoke policies

- **Stationary:** exact zero actions.
- **Paired random:** a complete `(H,M,2)` action tensor is generated before the episode from an independent policy stream and reused across null/signal pairs.
- **Privileged upstream oracle:** reads the true field only as a calibration diagnostic and moves against uniform transport. It is explicitly excluded from the primary estimand.

Three canonical 400-step smoke episodes executed without runtime errors. Their outcomes are not used as scientific comparisons.

### Dataset and reproducibility contract

Two closed JSON Schemas now define future trajectory-step and run-manifest records. The manifest requires repository revision, dirty-state flag, configuration hash, pairing declaration, policy identity and privilege flag, runtime versions and command, artifact byte counts, and SHA-256 digests. No trajectory writer or scientific dataset exists yet.

## Engineering verification

```text
python3 --version
python3 -c 'import numpy; print("NumPy", numpy.__version__)'
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m compileall -q src tests
python3 .../research_program.py check .
```

Verified environment:

- Python 3.12.13;
- NumPy 2.3.5;
- 44 tests passed;
- zero failures or errors;
- byte-compilation passed;
- canonical program-structure check passed.

### Retained failure history

The first provisional environment passed 36 tests, but root review found that the tests encoded the wrong scientific contract:

- collector positions were randomly initialized instead of frozen;
- newly visible particles received velocity information they could not have inferred causally;
- teammate information used the wrong observation rule;
- the privileged diagnostic moved with the field instead of upstream.

Those issues were corrected before publication. A later contract-edit run intentionally produced one failure and three errors because several old tests still expected the discarded behavior; those assertions were updated to the frozen contract without rolling back the implementation.

## Paper A status

**Question:** unchanged.  
**Claims:** all remain `proposed`.  
**Scientific results:** none.  
**Dataset:** schema only; no generated dataset.  
**Correctness pilot:** still `pilot_running`, because important gates remain.  
**Manuscript:** four pages, compiled successfully.  
**Current confidence:** high that the implemented engineering slice matches its declared contract; no confidence statement is made about the detectability hypothesis.

The strongest threat remains venue fit: a team of four collectors is not automatically a multi-agent contribution. The paper must eventually isolate a coordination effect beyond extra swept area and parallel search.

## Manuscript

The LaTeX manuscript now contains the implemented collector transition, deterministic reset, five random streams, observation tensor and causal velocity rule, exact step order, 44-test scope, trajectory/manifest schemas, and explicit remaining blockers. Stale text saying the environment and observation construction were absent was removed. Text implying an existing trajectory release was also removed.

Final compilation produced a four-page PDF with no undefined reference, layout overflow, or compilation warning. The source continues to state in bold that there is no scientific performance result.

## Literature

No new literature search was charged to this implementation cycle because the scientific question and novelty target did not change, and the 21-source ledger was refreshed earlier the same day. The nearest task-level threat remains Wang et al. (2025), which already studies locally guided mobile particle capture in flow. The nearest local-sensing RL threat remains Löffler et al. (2023). The active novelty target therefore remains exact counterfactual weak-field measurement plus a genuinely isolated multi-collector coordination mechanism.

## Fresh AAMAS review

**Recommendation:** 2/10 Reject  
**Confidence:** 4/5

The score did not improve because a conference reviewer evaluates scientific contribution, not the number of unit tests. The package still lacks `local_flow_v1`, a paired episode runner, dataset writer, estimator, signal sweep, boundary result, and independent-versus-sharing multi-agent comparison.

### Strongest accept path

A completed and reproducible paired measurement instrument, followed by evidence that a bounded shared summary changes the stable weak-signal crossing relative to an otherwise identical independent team after matching collector count, swept area, observation budget, policy capacity, action budget, and tuning budget.

### Strongest new technical threat

Capture is currently evaluated only at the end of a discrete step. A particle and collector can cross within the step without their endpoints being inside the capture radius. Since Brownian displacement is not obviously negligible relative to the capture radius, this can change the primary first-contact distribution. Performance pilots are blocked until the simulator either:

1. implements validated continuous segment contact, including reflected subpaths; or
2. freezes a timestep-convergence protocol demonstrating that the canonical discretization is adequate.

The review is preserved immutably at `paper/reviews/2026-07-31-post-environment-fresh-aamas.md`.

## Expansionist update

The event-keyed randomness entry now records the concrete active failure mode: stateful tie draws can become misaligned after paired trajectories diverge. It now includes a falsifiable consumption-order test and a bounded validation plan.

The aggregation entry now records that the primary environment uses point particles and fixed capture radius. Attached-node radius, aggregate-wall behavior, captured-particle motion, and geometry-matched controls must be frozen separately.

No industry document was changed because this cycle produced no new market evidence. Engineering progress alone is not evidence of customer demand, pricing, or commercial readiness.

## Blockers and HPC

No HPC is needed. The remaining work is correctness- and design-bound:

1. continuous contact or timestep convergence;
2. event-keyed tie breaking;
3. complete matched null/signal episode wrapper;
4. frozen `local_flow_v1` from causal observations only;
5. trajectory writer, schema-instance validation, hashes, and manifest generation;
6. growing attached-disc geometry contract;
7. only then, a small scripted baseline pilot.

## Autonomous next 24 hours

The next run will prioritize the two threats capable of invalidating first contact itself:

1. construct deterministic within-step crossing and reflected-path microcases;
2. choose and implement continuous contact or freeze a convergence study;
3. replace stateful capture-tie draws with event-keyed tie resolution;
4. build the matched episode wrapper and provenance audit;
5. freeze `local_flow_v1` only after the causal execution path passes;
6. implement trajectory writing and validate positive/negative schema examples;
7. rerun the entire suite and request another fresh review only after the scientific execution path materially changes.

## No-reply action

Continue the correctness-first sequence above. Do not start the signal sweep, MARL training, or growing-geometry analysis until the first-contact and matched-counterfactual blockers pass.
