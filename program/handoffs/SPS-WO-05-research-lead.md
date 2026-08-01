# SPS-WO-05 Research Lead Handoff

**Date:** 2026-07-31  
**Role:** Research Lead  
**Scope:** Paper A task-feasibility redesign only  
**Scientific status:** preregistered before any SPS-WO-05 performance run; SPS-C03 remains blocked

## Decision

Change exactly one element of the canonical task: replace first-interception time with **unique team capture yield after one sensing-radius traversal**, and do not stop the episode at the first contact. Keep the reset, particle count, physics, signal, and policies unchanged.

No SPS-P05 outcome was inspected or used in this decision. SPS-P05 remains permanently excluded from design selection, claims, power calculations, and manuscript interpretation.

## Why this is the defensible repair

SPS-P03 and SPS-P04 establish that the existing endpoint terminates before actions can accumulate useful leverage. Across their frozen diagnostic arms, every first contact was uncensored. In SPS-P04, the corrected oracle's median first contact was step 2 and its range was steps 1--8; stationary contact also occurred early (mean 5.78 steps across the grid). No wall-proximity event was present in the recorded pre-contact diagnostics. The corrected oracle could be absolutely earlier than stationary while still losing the signal-minus-null passive-adjusted contrast because its null first contact was already near the lower bound.

That diagnosis argues against buying more seeds or trying a stronger policy on the same first-event endpoint. It also does not justify changing density, clearance, speed, and horizon together. Counting all unique captures through a fixed movement-scale window removes the absorbing first-event floor while preserving the exact physical task.

The window is derived without looking at a redesigned-task outcome:

\[
L=\left\lceil\frac{r_{\mathrm{sense}}}{v_{\max}\Delta t}\right\rceil
=\left\lceil\frac{0.16}{0.12\times 0.02}\right\rceil=67.
\]

Thus the evaluation lasts 1.34 physical time units, enough for a max-speed collector to move one sensing radius. An incidental capture on step 1 remains recorded but cannot end the episode or alone determine the endpoint.

## Exact question

> Under the canonical four-collector reset and a uniform field with \(\alpha=0.06\) (\(\kappa=0.5\)), does the action-feasible full-state interception oracle capture at least one additional unique particle per collector, on average, than stationary collectors during one sensing-radius traversal window?

This is the only scientific question authorized by SPS-WO-05. It is a feasibility question, not a coordination claim. The bounded-sharing question remains blocked until this gate passes.

## Frozen task contract

Unchanged:

- reflected unit square, \(\Delta t=0.02\), diffusion \(\sigma=0.06\), and fixed capture geometry;
- 256 particles and four collectors at the canonical lattice positions;
- collector maximum speed 0.12, sensing radius 0.16, and capture radius 0.012;
- IID-uniform, capture-free particle initialization using the canonical stream;
- uniform field with the frozen orientation stream;
- exact within-step contact and event-keyed ties;
- matched initialization, Brownian tensor, field orientation, policy randomness, and tie provenance.

Changed:

- `stop_after_first_interception=false`;
- run through step 67;
- endpoint is the number of distinct particles captured by any collector through step 67.

No reset clearance or population-size change is permitted. This matters scientifically: a successful gate will be attributable to the endpoint's movement-scale integration, not to post-result geometry selection.

## Estimand and relevance threshold

Let

\[
Y_s(\pi,\alpha)=\text{number of unique particles captured by step 67}
\]

for seed \(s\), policy \(\pi\), and field strength \(\alpha\). The primary seed-level contrast is

\[
d_s=Y_s(\text{full-state oracle},0.06)-Y_s(\text{stationary},0.06),
\]

and the estimand is \(\theta=E_s[d_s]\). Freeze the practical relevance threshold at

\[
\theta_*=4,
\]

one additional unique capture per collector during the window. This threshold is an operational continuation criterion, not a universal statement of domain importance.

The mandatory supporting contrast is

\[
u_s=Y_s(\text{full-state oracle},0.06)-Y_s(\text{true-field upstream control},0.06).
\]

Report the matched \(\alpha=0\) outcomes for all three policies to expose passive capture and policy behavior without drift. They are supporting diagnostics. The primary estimand is deliberately a same-signal policy contrast, not the earlier signal-minus-null difference-in-differences that was structurally compressed by the first-event floor.

## Fresh diagnostic design

- Seeds: **2001--2008**.
- Policies: stationary, `true_field_upstream_control`, and `full_state_interception_oracle`.
- Conditions: \(\alpha\in\{0,0.06\}\); \(\alpha=0.06\) is primary and \(\alpha=0\) is diagnostic.
- Maximum workload: 48 policy-condition episodes, 67 steps each, or 3,216 environment steps.
- Hardware: Codex cloud CPU, at most 15 wall-clock minutes and 4 GiB RAM. No GPU, HPC, or learned-policy training.

For each policy-condition-seed cell, preserve unique yield, every capture step and owner, path length, matched-stream checksums, command, code revision, environment, runtime, and immutable gate output. Report all eight \(d_s\) and \(u_s\) values, their means and medians, and descriptive uncertainty. Eight seeds are a feasibility diagnostic, not a confirmatory significance test.

## Oracle gate

The gate passes only if all of the following hold:

1. the full test suite and deterministic oracle microcases pass, legal-action bounds hold, and the oracle has no future Brownian access;
2. \(\operatorname{mean}(d_s)\ge 4\);
3. \(d_s>0\) in at least 6 of 8 fresh diagnostic seeds;
4. \(\operatorname{mean}(u_s)>0\);
5. schemas, matched provenance, checksums, and immutable artifacts validate.

The one-per-collector threshold guards against declaring a scientifically tiny oracle advantage sufficient merely because it is positive. The seed-direction rule guards against a mean driven by one unusually large capture count. The true-field comparator requires target-aware full-state control to add something beyond simply moving against the field.

## Falsifier and stop rule

The redesign is falsified as a basis for the planned coordination study if any valid performance criterion above fails. Stop at the first correctness or provenance failure. One implementation-only repair is allowed if it leaves the endpoint, reset, policies, seeds, and thresholds unchanged; the rerun must use a new diagnostic experiment ID and preserve the failed artifact.

If a valid gate fails:

- keep SPS-C03 blocked;
- do not run shared versus independent control;
- do not run the timestep science slice or power study;
- do not train IPPO/MAPPO;
- do not try another endpoint, window, clearance, or particle count under this work order;
- return to a theory-led redesign decision or kill the active direction.

If the gate passes, it authorizes only the next diagnostic progression: coupled-noise timestep validation of this yield endpoint, followed by the already implemented bounded shared-summary versus capacity-matched independent controller. A pass is not evidence that communication works.

## Frozen interpretations

**Positive:** the oracle passes every gate. The unchanged canonical task has practically relevant action-contingent capture-yield headroom on this diagnostic slice. Proceed to numerical validation and a separately preregistered sharing diagnostic.

**Null:** the correct oracle is directionally nonnegative but misses the mean, consistency, or targeting threshold. This endpoint has not demonstrated enough action leverage for the proposed coordination study. Do not rescue it with more diagnostic seeds.

**Reversed:** stationary meets or exceeds the oracle on average, or the true-field-only control meets or exceeds it. Treat passive flux, target assignment, or action-induced opportunity loss as the leading explanation and stop communication experiments.

**Invalid:** a test, leakage, pairing, schema, provenance, or checksum condition fails. Repair only that implementation defect, preserve the failed run, and repeat the identical frozen design.

## Diagnostic/confirmation firewall

Seeds 2001--2008 and all outcomes produced under SPS-WO-05 are permanently diagnostic. They may inform a later power calculation but must never be pooled into a confirmatory estimate or cited as claim evidence. Any later confirmation must preregister a disjoint seed set excluding the calibration and diagnostic seeds used by SPS-P01 through SPS-P04 and SPS-WO-05. The endpoint, window, alpha, policies, threshold, and gate freeze before the first SPS-WO-05 outcome is viewed.

## Engineer's next bounded action

Implement and test the step-67 unique-yield endpoint without touching reset or policy behavior. Required deterministic tests are: unique counting; inclusion of step-67 events; exclusion of step-68 events; continued simulation after the first contact; and exact matched-stream provenance. Then run only the frozen oracle diagnostic above. No manuscript result, coordination experiment, or HPC request is authorized by this handoff.
