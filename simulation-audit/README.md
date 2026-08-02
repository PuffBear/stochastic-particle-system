# Simulation V&V Audit for Event-Driven Systems

**Industry idea:** FI-3
**Status:** Unvalidated — internal P02 null result is a reason for caution, not urgency
**Readiness:** Low

## Problem

Simulation-based engineering conclusions that depend on first-hit, first-failure, or threshold-crossing events are rarely tested for sensitivity to timestep choices or event-handling conventions — even when those conclusions support safety, regulatory, or procurement decisions.

The SPS project found this problem directly: the original environment produced zero observed event ties in first-interception. Without the SPS-P02 audit, this structural property would never have been detected and verified. The audit found the environment was robust to that specific change — but the audit is what made that claim credible.

## The audit concept

Given a simulation trace with logged events, replay under alternative numerical assumptions and check whether:
- Event timestamps shift under alternative timestep choices
- Event ownership changes under alternative contact-detection models
- Policy rankings reverse under alternative tie-resolution conventions

**Output:** A provenance certificate — a structured document showing which numerical assumptions each reported conclusion survived, suitable for V&V reports or regulatory submissions.

## Honest status

The strongest internal evidence is the SPS-P02 null result: changing exact contact mathematics changed 0/144 first-interception outcomes. This demonstrates SPS robustness, not a commercial problem. Before building, external evidence is needed that event-handling sensitivity is a real, unresolved problem in at least one target industry.

## Files

- `business-questions.md` — target industries, problem evidence needed, validation plan, revenue model, constraints
