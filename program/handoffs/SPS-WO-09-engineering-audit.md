# SPS-WO-09 engineering feasibility audit

**Repository snapshot:** `8ea01cecc5391e7e317943931a75447a64c99f99`  
**Audit type:** read-only code and deterministic-test audit; no scientific seed
run.

## Reusable platform components

- fixed-geometry capture, unique ownership, and unique-yield outcomes;
- local particle positions and causally valid apparent velocities;
- three-scalar `(vx,vy,valid fraction)` local summaries;
- all-to-all arithmetic summary and capacity-matched local controller;
- deterministic field/noise/tie seed streams and matched provenance checks;
- trajectory actions/positions/captures plus offline distance, sensing-overlap,
  and coverage diagnostics.

Fifty deterministic core/environment/policy tests passed during the audit.

## Missing components that block SPS-C04

1. No stationary field with a defined correlation length. `vortex_scale` is
   only a radial envelope width.
2. No pure receiver-specific communication graph or explicit link/broadcast
   accounting.
3. The current global controller sends the same mean, including self, to every
   receiver and cannot log receiver-specific messages.
4. Teammate positions are currently available in observations and must be
   disabled or controlled for the primary intervention.
5. No message, adjacency, aggregation-weight, true-field error, or action-
   similarity artifact exists.
6. The current paired runner compares signal and null under one policy; it is
   not a matched communication-arm runner.
7. Immutable trajectory schema v1 should not be silently expanded; add a new
   message-diagnostic schema.

## Implementation order

1. add and validate the declared-covariance episode-frozen field;
2. add a pure `(M,3)` message aggregation primitive with explicit adjacency and
   self-inclusion;
3. add independent/all-to-all capacity-matched adapters;
4. add constant, opposing-region, homogeneous-noise, radius-limit,
   permutation, exact-bandwidth, and no-valid-evidence tests;
5. add message/mechanism diagnostics under a new schema;
6. add a matched communication-arm runner reusing initialization, Brownian
   tensor, field realization, and tie randomness;
7. only then design a fresh bounded performance diagnostic.

No HPC is required for this sequence.
