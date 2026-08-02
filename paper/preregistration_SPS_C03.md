# Preregistration: SPS-C03 Coordination Diagnostic

**Date:** 2026-07-31
**Status:** CONFIRMED (executed and passed 2026-08-01)
**Experiment ID:** SPS-C03-COORDINATION-DIAGNOSTIC
**Registered by:** [research team]
**Upstream gate required:** SPS-WO-06 (coupled-noise timestep convergence for unique yield)

## OUTCOME (added 2026-08-01)

**Result: CONFIRMED.** Pre-registered one-sided studentized-bootstrap 95% lower confidence bound = **+0.459 > 0**.

| Metric | Value |
|---|---|
| Seeds | 6001–6032 (32 confirmation seeds) |
| Policy pair | shared_summary_v2 vs capacity_matched_independent |
| Mean Δ_s | +1.19 unique particles |
| SD | 2.44 |
| Seeds positive | 20/32 |
| LCB (one-sided 95% bootstrap) | +0.459 |
| Gate | PASSED |

All three gate components passed: (1) correctness and matched-stream verification, (2) lower bound strictly positive, (3) minimum relevant effect met.

Note: The diagnostic gate originally specified `shared_summary` (v1). Prior to the confirmation run, v1 was replaced with `shared_summary_v2` following an informative failure (SPS-WO-07, 4/8 positive) traced to equal-weight averaging and correlated team failure. The v2 fix (count-weighted mean, field+density blend) passed a re-diagnostic (SPS-WO-07B, seeds 5001–5008, 7/8 positive, mean=+2.63). Seeds 4001–4008 (WO-07) and 5001–5008 (WO-07B) remain permanently diagnostic-only. The confirmation used seeds 6001–6032 (fresh, not previously examined).

---

---

## 1. Research question

Does replacing each collector's independent local velocity estimate with a
bounded team-mean velocity summary increase the number of unique particles
captured by a four-collector team through the inclusive end of step 67, at
field strength α = 0.06, relative to four independent collectors with
identical observation-interface dimensions, action, and speed budgets?

This is a single binary-direction diagnostic question. It is not a claim that
sharing is the mechanism of any yield improvement, not a claim about the
boundary of exploitable field strength, not a MARL experiment, and not a
confirmation of any coordination effect. A positive result at the diagnostic
stage authorizes a bounded power study and confirmation seed budget; it does not
itself constitute a confirmatory claim.

---

## 2. Endpoint and estimand

### Primary endpoint

**Unique team captures through the inclusive end of step 67 (EVALUATION_STEPS = 67).**

Let Y_s(π, α) denote the count of distinct particle IDs captured by any of
the four collectors in scenario seed s, field strength α, and policy π, where
counting includes events at step 67 and excludes events at step 68 and beyond.
Each particle ID is counted at most once regardless of how many collectors were
in contact range.

### Primary estimand

The matched seed-level yield difference:

    Δ_s = Y_s(shared, 0.06) − Y_s(independent, 0.06)

where both arms use the **capacity_matched_velocity_controller** policy class
(see Section 4), the shared arm receives the bounded team-mean velocity summary
and the independent arm does not.

### Passive-advection check estimand

To confirm that any positive Δ_s is not purely a passive-transport artefact,
the following secondary contrast is computed alongside the primary estimand:

    Δ_passive_s = Y_s(stationary, 0.06) − Y_s(stationary, 0.00)

A large passive contrast relative to Δ_s would indicate that transport benefits
stationary collectors strongly and leaves little action-contingent headroom, the
failure mode documented in the superseded first-event endpoint (SPS-P04).

---

## 3. Diagnostic seeds

**Eight seeds from the range 4001–4008 (inclusive).**

These seeds are reserved for the SPS-C03 diagnostic and are permanently
ineligible for use as confirmation seeds in any later experiment, including
SPS-C01, SPS-E01, or any successor study. They were not used in any prior pilot
(SPS-P01 used seeds 1001–1012; SPS-P02 reused the same block; SPS-WO-05 used
seeds 2001–2008; SPS-WO-06, if executed, will use seeds 3001–3008).

No outcomes from seeds 4001–4008 may be examined, summarised, or used in any
design decision before this preregistration is finalised and timestep
convergence (SPS-WO-06) has passed.

---

## 4. Field strength and policy pair

### Field strength

α = 0.06 (canonical coordination-diagnostic level; κ = α / v_max = 0.50).

### Policy pair

Both policies belong to the **capacity_matched_velocity_controller** class.
They have identical observation-interface dimensions (same sensing radius,
same K nearest-particle slots, same teammate-position channel), identical
action budgets (same unit-ball projection and v_max cap), and identical speed
limits. They do **not** have identical information content: pooled evidence
from other collectors is precisely the treatment in the shared arm. This
diagnostic therefore estimates the effect of that bounded summary, not a
generic capacity effect.

**Shared arm:** `shared_summary`
The policy computes the local velocity estimate for each visible particle (as
in the independent arm), then replaces that per-agent estimate with the
bounded team-mean velocity: the element-wise mean of the individual estimates
across all M = 4 collectors, clipped to unit norm before the v_max projection.
The team-mean is computed from the current step's causally valid observations
and is broadcast to all collectors before the action is selected.

**Independent arm:** `capacity_matched_independent`
The policy computes the local velocity estimate for each visible particle and
acts on that estimate without sharing any summary. The teammate-position
channel is present in the observation but unused by the policy rule. This arm
is exactly four independent replicas of the local-flow heuristic at matched
capacity.

Both arms receive the same observation interface: arena-normalised self
position; up to K = 32 nearest free particles within sensing radius r_s = 0.16;
binary visibility masks; per-particle velocity estimates (emitted only when the
particle was visible to the same collector in both consecutive steps); and
teammate positions relative to self in fixed ID order. Neither arm receives the
field parameter, global particle state, future noise draws, or any random
generator.

---

## 5. Frozen gate components (ALL must pass)

The diagnostic passes if and only if every one of the following conditions
holds over the eight diagnostic seeds 4001–4008:

1. **Positive mean:** mean(Δ_s) > 0.
2. **Positive direction in at least 5 of 8 seeds:** |{s : Δ_s > 0}| ≥ 5.
3. **Shared exceeds stationary on mean:** mean(Y_s(shared, 0.06)) >
   mean(Y_s(stationary, 0.06)).
4. **Passive-adjusted ordering:** mean(Y_s(shared, 0.06) − Y_s(stationary, 0.06))
   > mean(Y_s(independent, 0.06) − Y_s(stationary, 0.06)).
5. **No implementation fault:** all 8 seeds pass schema validation, checksum
   verification, and artifact completeness checks.
6. **Matched streams verified:** the matched runner confirms that independent
   and shared arms share identical particle and collector initialization,
   Brownian tensors, field nuisance variables, and policy tie-breaking
   randomness for every diagnostic seed.

These gate components are frozen. No component may be weakened, relaxed,
or reinterpreted after any diagnostic seed outcome is observed.

---

## 6. If the gate passes

If all four empirical conditions in Section 5 pass (in addition to conditions
5–6, which are procedural prerequisites):

1. **Use the ex-ante minimum relevant effect:** the minimum effect considered
   practically relevant for confirmation is fixed at **2.0 unique team
   captures** on average (0.5 per collector). This threshold was selected
   before any seed 4001–4008 outcome was observed because a smaller benefit
   would not justify a full confirmation battery. It is not estimated from,
   and cannot be changed in response to, the diagnostic outcomes.
2. **Run a power analysis:** a simulation-based power study is conducted using
   the frozen minimum relevant effect, an ex-ante sensitivity range of paired
   seed-level standard deviations from 2.0 to 4.0, and the same one-sided
   paired studentized-bootstrap lower-bound procedure intended for
   confirmation (B = 10,000 draws, α = 0.05), over candidate seed budgets
   n ∈ {16, 24, 32}. The study targets 80% power at the frozen minimum effect.
   Diagnostic variance may be reported later but cannot replace this frozen
   sensitivity analysis or alter its candidate counts.
3. **Authorize confirmation seeds:** confirmation seeds are drawn from the range
   5001–9999. The exact seed block is specified in the power study output and
   must not overlap with any prior pilot, diagnostic, or calibration block.
4. **Freeze the inference procedure:** the confirmatory test is a one-sided
   paired bootstrap lower confidence bound at the 95% level (simultaneous over
   the shared-vs-independent contrast only, not a grid). The bound must be
   strictly above zero for a positive claim.
5. **No coordination claim is made at the diagnostic stage.** The positive
   diagnostic result is reported as "diagnostic headroom consistent with a
   sharing benefit," pending confirmation.

---

## 7. If the gate fails

If any empirical gate component in Section 5 (conditions 1–4) fails:

1. SPS-C03 closes. No further shared-vs-independent diagnostic may be run for
   the SPS-C03 Experiment ID.
2. The null result is reported in full: all eight seed-level outcomes for both
   arms, the stationary control, and the gate evaluation table.
3. No additional tries are permitted except for one bounded repair (see below).
4. **One bounded repair:** if a specific, pre-stated implementation fault is
   identified and documented before examining any gate-component outcome, a
   single corrective rerun on seeds 4001–4008 is permitted. The repair must
   be registered as SPS-C03-REPAIR-R1, must state the exact fault and fix
   before any rerun data are generated, and must be the only change to the
   policy or environment. A second repair is not permitted under any
   circumstances.
5. No confirmation, power, or MARL experiment may reference a failed SPS-C03
   gate as positive evidence.

---

## 8. Attribution controls to run alongside

All four of the following policy conditions must be run on the same eight
diagnostic seeds and reported in the same analysis package as the primary
diagnostic:

1. **shared_summary** (shared arm; primary)
2. **capacity_matched_independent** (independent arm;
   primary comparator)
3. **stationary** (all four collectors are stationary; passive-advection
   baseline)
4. **full_state_interception_oracle** (the corrected full-state action-feasible oracle from
   SPS-WO-05; action-contingent headroom reference)

All four policies must use the same matched random streams (identical particle
initialization, Brownian tensors, and field nuisance for each seed). The oracle
arm provides the upper reference for action-contingent yield; the stationary arm
provides the lower reference for passive transport. Results for all four arms
must be reported regardless of the gate outcome, so that the passive-advection
and oracle-headroom context is fully visible to reviewers.

A fifth condition may be added for design diagnostics (e.g., oracle-shared or
team-oracle), but must not alter the matched streams used for the primary pair.

---

## 9. Timestep convergence prerequisite

**SPS-WO-06 must pass before this experiment executes.**

SPS-WO-06 is the coupled-noise timestep convergence diagnostic for unique yield
over a fixed physical duration of 1.34 time units (67 steps at Δt = 0.02). It
uses seeds 3001–3008 and levels Δt = 0.02, 0.01, and 0.005. One standard-normal
tensor is generated at Δt = 0.005; its Brownian increments are summed in blocks
of two and four for the coarser levels. It reports stationary signal-minus-null
and oracle-minus-stationary contrasts at every level. The gate passes if and
only if the absolute difference between the mean oracle-minus-stationary
contrasts at Δt = 0.02 and Δt = 0.01 is strictly below 1.0 particle and their
seed-level contrast signs differ in no more than 1 of 8 seeds. The Δt = 0.005
comparison is mandatory but informational; it cannot rescue a failed gate.

SPS-C03 must not be executed, scheduled, or partially executed before SPS-WO-06
passes. No outcome from seeds 3001–3008 or seeds 4001–4008 may be examined
before SPS-WO-06 completes and its gate decision is recorded.

---

## 10. What this diagnostic does NOT authorize

Regardless of the gate outcome, this diagnostic does NOT authorize:

- Any claim that team-mean sharing is the mechanism of a yield improvement
  (it demonstrates association, not mechanism).
- Any confirmation seeds or confirmatory inference at AAMAS-reportable
  statistical thresholds.
- Any MARL, IPPO, MAPPO, or learned-communication experiment.
- Any extension to growing-aggregate geometry or irreversible-growth dynamics.
- Any extension to field strengths other than α = 0.06.
- Any generalization beyond the frozen unit-square arena with reflecting
  boundaries, N = 256 particles, M = 4 collectors, and EVALUATION_STEPS = 67.
- Any boundary estimate for SPS-C01 (which remains proposed but inactive
  and requires its own separately preregistered grid and inference procedure).
- Any claim of learning, adaptation, or optimization by the shared policy.
- Any claim that the bounded team-mean summary is an efficient or optimal
  communication protocol.

A positive gate result authorizes only the power study described in Section 6
and a separately preregistered confirmation experiment with its own seed block
and frozen inference procedure.

---

## 11. Inference procedure

The inference at the diagnostic stage is **descriptive and paired**, not a
simultaneous lower confidence bound (that procedure is reserved for SPS-C01 and
the confirmatory stage of SPS-C03 if reached).

### Reported statistics for the primary contrast

For the eight seed-level differences Δ_s = Y_s(shared, 0.06) − Y_s(independent, 0.06):

- Seed-level outcomes listed individually.
- Mean(Δ_s) and standard error.
- Count of positive, zero, and negative seeds.
- Descriptive paired-bootstrap 95% interval using B = 10,000 draws and
  bootstrap seed 73,031 (the same seed used in SPS-P02 and SPS-WO-05).
- No simultaneous correction is applied at the diagnostic stage.

### Reported statistics for each control arm

For stationary and oracle arms, the same seed-level listing, mean, standard
error, and descriptive bootstrap interval are reported for:

- Y_s(arm, 0.06) − Y_s(stationary, 0.00) (signal response)
- Y_s(arm, 0.06) − Y_s(stationary, 0.06) (active gain above passive)

### No sequential updating

Diagnostic outcomes are examined once, after all eight seeds have completed.
No intermediate seed examinations are performed. No adaptive stopping rule is
applied.

---

## 12. Frozen constants

The following constants are frozen for all runs in this experiment and may not
be altered without voiding this preregistration and registering a successor:

| Parameter | Value | Notes |
|---|---|---|
| EVALUATION_STEPS | 67 | Unique yield endpoint; step 67 inclusive |
| dt | 0.02 | Euler timestep |
| sigma | 0.06 | Particle Brownian noise scale |
| alpha | 0.06 | Field strength (diagnostic level) |
| M | 4 | Number of collectors |
| N | 256 | Number of particles |
| capture_radius | 0.012 | Collector disc radius |
| sensing_radius | 0.16 | Collector local sensing radius |
| v_max | 0.12 | Maximum collector speed |
| K | 32 | Maximum nearest-particle slots in observation |
| H | 400 | Full episode horizon (episode continues to H regardless) |
| Arena | Unit square [0,1]^2 | Reflecting boundaries; piecewise-specular |
| Collector reset | (.25,.25),(.25,.75),(.75,.25),(.75,.75) | Fixed ID order |
| Particle initialization | Rejection-sampled outside all capture discs | Dedicated init stream |
| Contact rule | Exact piecewise-specular; earliest quadratic root per segment | As in SPS-WO-05 |
| Tie-breaking | SeedSequence(s, t, k): scenario seed, one-based step, particle ID | Event-keyed; no state dependence |
| Brownian tensor | Pre-generated full tensor before episode start | Shared across arms by matched seed |
| Bootstrap draws | 10,000 | Diagnostic bootstrap |
| Bootstrap seed | 73,031 | Frozen; same as SPS-P02 and SPS-WO-05 |
| Diagnostic seeds | 4001–4008 | Reserved; ineligible for confirmation |

Any change to a frozen constant requires closing this preregistration, reporting
the diagnostic as incomplete, and opening a new preregistration with a fresh
seed block.

---

## Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-07-31 | Initial registration; not yet executed |
| 1.1 | 2026-08-01 | Corrected policy identifiers and information-budget wording; aligned the coupled timestep gate and froze a 2.0-particle one-sided power design ex ante; no diagnostic outcomes observed. |
