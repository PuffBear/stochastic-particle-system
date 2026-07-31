# SPS-WO-04 Research Lead Handoff

**Date:** 2026-07-31  
**Role:** Research Lead  
**Scope:** Paper A only; frozen fixed-geometry stochastic-particle benchmark  
**Evidence inherited:** SPS-P00 engineering gate, SPS-P01 exploratory calibration, SPS-P02 exact-contact same-seed replication, and the 2026-07-31 compressed-week fresh AAMAS review  
**Scientific status:** the instrument is engineered but the proposed mechanism is not established. SPS-P02 is preserved as negative exploratory evidence and must not be overwritten, pooled into confirmation, or re-labelled as support.

## Decision in one sentence

Do not run SPS-E01 or train MARL. First test whether a full-state interception controller can produce a policy-dependent improvement beyond passive transport in a catchable regime; only if that gate passes should the program test the genuinely multi-agent claim that one bounded shared velocity summary improves signal use over an information- and capacity-matched independent policy.

## Smallest unresolved scientific question

> At fixed physical dynamics with particle drift no faster than the collectors, does a bounded team velocity summary improve passive-adjusted, horizon-censored first-interception time relative to the same scripted controller using independent local velocity estimates?

This is one question. The estimator, oracle, timestep, and power tasks below are validity gates for answering it, not additional research questions.

The phrase **passive-adjusted** means that the signal/null improvement of a policy is differenced against the signal/null improvement of stationary collectors. For scenario seed \(s\), policy \(\pi\), and signal strength \(\alpha\), define

\[
G_s(\pi,\alpha)=\frac{T_{s,\pi,0}-T_{s,\pi,\alpha}}{H},
\qquad
A_s(\pi,\alpha)=G_s(\pi,\alpha)-G_s(\text{stationary},\alpha),
\]

where \(T=H+1\) when first contact is censored. The eventual coordination contrast is

\[
\Gamma_s(\alpha)=A_s(\text{shared},\alpha)-A_s(\text{independent},\alpha).
\]

Because the same stationary term cancels algebraically in \(\Gamma\), it is retained and reported to diagnose passive flux, while the primary coordination estimator is equivalently \(G_s(\text{shared})-G_s(\text{independent})\). Absolute null and signal outcomes must still be reported; a contrast alone is insufficient.

## Active claim and falsifier

**Active claim:** SPS-C03, narrowed to the frozen scripted mechanism: under identical observation dimension, action limits, collector count, compute, and policy capacity, replacing independent per-agent local velocity aggregation with one bounded shared team velocity summary yields a positive mean coordination contrast \(E[\Gamma_s(\alpha)]\) at at least one preregistered catchable signal point, with persistence at stronger catchable points.

**Falsifier:** the claim is falsified for the frozen task if any of the following occurs:

1. the full-state interception oracle fails the oracle-feasibility gate after bounded diagnostic repair;
2. the apparent signal effect is explained by stationary transport, swept area, or shuffled velocity slots;
3. shared-summary minus independent performance is non-positive, is smaller than the program-relevance threshold, or disappears under equalized information/capacity controls;
4. the result changes materially under the two finest coupled-noise timesteps;
5. simulated power for the smallest relevant effect requires more than 64 independent seeds under the bounded CPU design.

SPS-C01 should remain **proposed but inactive**. SPS-P02 neither supports nor rejects it because those seeds were calibration-only, and the current `local_flow_v1` treatment is confounded by passive transport and lacks a multi-agent mechanism.

## Why this is the highest-information next question

The present uncertainty is structural, not merely statistical. More seeds on `local_flow_v1` would precisely estimate a contrast that may mean only passive advection. The ordered gates below distinguish five mutually actionable failure modes:

- **oracle fails:** the physical task or endpoint is not exploitable; redesign or kill the canonical task;
- **oracle passes but local estimator fails:** repair sensing/estimation, not the simulator or sample size;
- **local controller passes but sharing does not:** retain a single-agent benchmark or pause AAMAS positioning;
- **sharing passes but timestep does not:** repair numerical approximation before inference;
- **all mechanism gates pass but power exceeds the cap:** narrow the claim or endpoint instead of spending compute.

Thus every bounded run changes a decision. No run is authorized merely to accumulate positive-looking plots.

## Frozen physical axes for the repair round

The old \(\rho=\alpha\sqrt{\Delta t}/\sigma\) axis must be reported but may not be the sole axis. Also report

\[
\kappa=\alpha/v_{\max},
\]

which separates physical catchability from per-step drift-to-Brownian strength. Under the current \(\Delta t=0.02\), \(\sigma=0.06\), and \(v_{\max}=0.12\), the old \(\rho=2\) point has \(\kappa\approx7.07\), so advection is much faster than collector motion.

For the bounded repair pilot, freeze the catchable grid

- \(\kappa\in\{0,0.25,0.5,1.0\}\), hence \(\alpha\in\{0,0.03,0.06,0.12\}\);
- report corresponding \(\rho\) at each timestep rather than holding \(\rho\) fixed during refinement;
- arena, particle count, collector count, radii, sensing radius, diffusion, initialization, and physical horizon remain canonical;
- calibration seeds 1001--1012 may be reused only for diagnostics and power-model construction, never confirmation.

The upper point \(\kappa=1\) is a feasibility boundary, not evidence that lower signal strengths are detectable.

## Diagnostic hierarchy and bounded preregistered work order

### Gate 0 — preserve and reproduce the negative evidence

**Can be done now.**

- Verify checksums and the one-command regeneration path for SPS-P02 summaries.
- Add no new interpretation to SPS-P02: all simultaneous lower bounds were negative; stationary and random descriptively matched or exceeded local flow; exact contact changed 0/144 first-contact outcomes.
- Record the existing one-collector/four-collector reversal as unresolved, not as a coordination effect.

**Stop condition:** any checksum or reproduction mismatch blocks all new evidence runs until repaired.

### Gate 1 — instrument policy and passive-transport diagnostics

**Can be implemented and smoke-tested now; run only on diagnostic seeds.**

For every collector-step, record without changing the policy:

- visible-particle count and causally valid velocity-track count;
- fraction of steps with at least one valid track and fraction with nonzero action;
- estimated mean local velocity, true latent field velocity, angular error, and magnitude error;
- action alignment with estimated velocity, true field, and nearest visible-particle direction;
- distance to nearest wall and a wall-proximity indicator;
- collector path length, swept-area proxy, first-contact owner, absolute first-contact time, censoring, and restricted mean first-contact time;
- the same summaries by null/signal arm, agent, seed, and \(\kappa\).

Mandatory diagnostic controls at each nonzero \(\kappa\): stationary; pregenerated random; deterministic coverage; density greedy; `local_flow_v1`; local flow with valid velocity slots shuffled across time within agent and seed; local flow with velocity-slot agent labels permuted; action-sign reversal; and local flow with a density fallback when no track is valid.

**Primary Gate-1 diagnosis:** report \(A_s(\pi,\alpha)\) plus absolute outcomes. `local_flow_v1` is mechanism-feasible only if it exceeds stationary and both velocity-shuffled controls in the expected direction. Descriptive ranking is not confirmation.

**Bound:** use seeds 1001--1008 initially and at most 8 seeds × 4 grid points × 10 policies. Stop as soon as a deterministic correctness failure appears. These are exploratory diagnostics.

### Gate 2 — replace the weak oracle

**Can be implemented and microcase-tested now; bounded pilot follows Gate 1 instrumentation.**

Keep `privileged_upstream_oracle` as a true-field-only control, but do not call it an interception oracle. Add a distinct centralized full-state interception oracle that, each step:

1. observes all free particle positions and their true current motion state;
2. predicts feasible constant-velocity intercepts over a bounded receding horizon under the same collector speed and boundary rules;
3. assigns distinct particle targets to collectors by minimum predicted intercept time;
4. emits only legal two-dimensional actions and receives no future Brownian increments.

The implementation may use deterministic exhaustive assignment over a bounded candidate set; it must not depend on a learned model or future noise. Unit microcases must show that it reaches an interceptable moving target earlier than stationary and does not claim an intercept when target speed/geometry makes one impossible.

**Oracle-feasibility gate:** on diagnostic seeds at \(\kappa\in\{0.25,0.5,1\}\), the full-state oracle must have positive mean passive-adjusted gain and outperform the true-field-only, stationary, and swept-area controls in absolute restricted mean first-contact time. Require consistent direction in at least 6 of 8 seeds at one grid point before proceeding. This is a cheap feasibility rule, not a significance test or manuscript claim.

**Failure action:** if the oracle fails, inspect endpoint saturation, wall accumulation, initialization, and horizon once. Permit one bounded repair that changes only a preregistered task parameter. If it still fails, kill or redesign the canonical first-interception question; do not proceed to sharing or MARL.

### Gate 3 — coupled-noise timestep convergence

**Implementation can begin now; the convergence run should follow a passing oracle microcase.**

- Hold physical horizon, \(\alpha\), \(\sigma\), initial state, collector speed, and geometry fixed.
- Use \(\Delta t\in\{0.02,0.01,0.005\}\).
- Generate the finest Brownian increments first and obtain coarser increments by exact summation, so paths are coupled across timesteps.
- Evaluate stationary, full-state oracle, independent local flow, and the eventual shared-summary policy.
- Report first-contact disagreement, absolute restricted mean first-contact time, passive-adjusted effects, and shared-minus-independent effects.

The endpoint is provisionally stable only if the mean normalized contrast changes by at most 0.0025 between \(0.01\) and \(0.005\), no policy ordering reverses, and the interval around the refinement difference excludes changes larger than the eventual 0.01 relevance threshold. With only diagnostic seeds, report uncertainty rather than declaring convergence.

**Bound:** 8 diagnostic seeds × 3 nonzero \(\kappa\) × 3 timesteps × at most 4 policies. If the finest run is unexpectedly expensive, finish the two-policy stationary/oracle slice first. No HPC.

### Gate 4 — freeze the genuinely multi-agent mechanism

**Design and unit tests can begin now; scientific pilot is tomorrow's progression after Gates 1--3.**

Define exactly one shared channel: a two-dimensional clipped team mean of agents' locally estimated particle velocities plus a one-dimensional validity fraction. Each agent receives the same three-number message. No particle identities, global positions, latent field, future noise, or extra history may enter the channel.

Freeze two controllers with identical deterministic functional form and parameter count:

- **independent:** consumes its own clipped two-dimensional velocity estimate and own validity fraction;
- **shared:** replaces those three inputs with the bounded team aggregate.

Both receive the same self-position and teammate-position inputs already allowed, use the same fallback, normalization, action limit, action frequency, and arithmetic budget. To equalize observation dimensionality, the independent controller's three local values occupy the exact message slots used by the shared controller. Add message-shuffled and leave-one-agent-out ablations. Verify permutation invariance of the team summary and agent-ID equivariance of actions.

This is the minimum defensible AAMAS mechanism: bounded communication changes evidence fusion while action capacity and policy form stay fixed.

### Gate 5 — simulation-based power and confirmation authorization

**Do after the diagnostic distributions and timestep gate exist.**

Use a program-relevance threshold of \(\delta_*=0.01\) in normalized horizon units (four canonical steps, or 0.08 physical time units) for the coordination contrast. This is an operational continuation threshold, not a claim of universal domain importance.

- Fit no parametric success story to the 12 calibration seeds.
- Resample complete scenario-seed blocks preserving censoring, policy correlation, and cross-grid dependence.
- Simulate null, reversed, and local alternatives centered at \(0.5\delta_*\), \(\delta_*\), and \(1.5\delta_*\).
- Require familywise type-I error at most 0.05 (simulation upper 95% bound at most 0.07) and at least 80% power at \(\delta_*\).
- Candidate independent seed counts: 16, 24, 32, 48, 64; choose the smallest passing count before seeing confirmatory outcomes.
- Use a one-sided simultaneous 95% lower bound across the three nonzero catchable points. Call a boundary only if the crossing persists at all stronger preregistered points; otherwise report the full non-monotone curve and no boundary.

**Compute kill:** if 64 seeds do not achieve 80% simulated power at \(\delta_*\), do not run confirmation. Narrow the endpoint or pause SPS-C03. Do not request HPC for an underidentified or underpowered mechanism.

## Interpretations frozen before the pilot

**Positive:** the oracle, timestep, and power gates pass; the shared controller has a positive simultaneous lower bound and at least \(\delta_*\) mean coordination contrast at a persistent catchable point; stationary and shuffled controls cannot explain it. This supports only the frozen bounded-summary mechanism under the frozen task.

**Null:** oracle and timestep gates pass, but sharing does not exceed the independent controller. Conclude that this bounded summary does not improve the chosen interception endpoint; do not conclude communication is generally useless.

**Reversed:** independent local estimates outperform the shared summary or shuffled messages match sharing. Treat aggregation dilution, nonstationary local geometry, or message-induced correlated actions as primary explanations; pause the coordination claim.

**Invalid:** oracle failure, passive-control explanation, provenance mismatch, timestep instability, information leakage, or inadequate power. Repair only the failed upstream gate; do not interpret performance.

## What should be completed now

1. Preserve and checksum SPS-P02; add its negative result to the work-order preamble.
2. Implement diagnostics and their schemas/tests without changing `local_flow_v1`.
3. Implement and microcase-test the distinct full-state interception oracle while retaining the upstream policy under its accurate name.
4. Implement coupled Brownian refinement utilities and deterministic aggregation tests.
5. Implement the bounded shared-summary and identical-shape independent controller, with permutation/equivariance and leakage tests.
6. Run only bounded smoke and 8-seed diagnostic slices in dependency order. Stop downstream execution if the oracle does not clear feasibility.
7. Produce simulation-based power code and synthetic calibration tests, but do not authorize independent confirmation yet.

## Tomorrow's progression

If today's oracle and diagnostic gates pass, tomorrow should:

1. run the coupled-noise timestep slice and decide whether \(\Delta t=0.02\) is usable or must be replaced;
2. run the frozen 8-seed shared-versus-independent diagnostic pilot on the catchable grid;
3. evaluate passive-adjusted, shuffled-message, and leave-one-agent-out contrasts;
4. construct the empirical block distribution and choose or reject a confirmatory seed count via simulation-based power;
5. update SPS-C03 and preregister a new experiment ID only if every upstream gate passes;
6. commission a fresh AAMAS review of the resulting immutable package.

If today's oracle gate fails, tomorrow should not test coordination. It should execute the single permitted bounded task repair, rerun the oracle diagnostic, and then kill or redesign the first-interception task if feasibility still fails.

## Explicit prohibitions

- Do not run the blocked 24-seed SPS-E01.
- Do not reuse seeds 1001--1012 for confirmation.
- Do not describe the upstream true-field policy as a full-state oracle.
- Do not infer active exploitation from signal versus null without passive and shuffled controls.
- Do not infer coordination from four agents running independent replicas.
- Do not hold \(\rho\) fixed across timestep refinement while silently changing physical drift.
- Do not train IPPO/MAPPO, request HPC, or activate a second paper before the scripted mechanism, numerical stability, and power gates pass.

## Expected information gain

This work order should end with one of four decision-grade outcomes: **task infeasible** (oracle fails), **estimator/policy infeasible** (oracle passes but causal local velocity use does not), **coordination mechanism unsupported** (local use passes but sharing does not), or **confirmation justified** (all gates and bounded power pass). Each outcome removes a major ambiguity left by SPS-P02; none converts its negative calibration into positive evidence.
