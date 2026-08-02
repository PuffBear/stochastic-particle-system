# Coordination Diagnostic Tool

**Industry idea:** FI-1
**Status:** Unvalidated — interview validation required before building
**Readiness:** Medium

## Problem

Engineering teams building distributed sensing or collection systems have no systematic way to answer: "Is our communication protocol helping or hurting?" Standard simulators report aggregate reward; they cannot tell you whether the *content* of communication is doing causal work or simply adding correlated noise.

The SPS WO-07 result demonstrated this failure mode: equal-weight sharing made performance worse than no communication on 4/8 seeds. This failure would be invisible in standard simulator output — total reward decreased, but nothing would surface that communication content was the cause.

## Product concept

A simulation or replay analysis tool that runs a team's scenario under three conditions:
1. **Actual messages** — the team communicates as designed
2. **Permuted messages** — same format and frequency, content randomly shuffled across agents each step
3. **No messages** — agents receive a zero-filled slot

**Output:** A three-number diagnostic:
- Gain from content: comparison 1 vs 2
- Gain from structure: comparison 2 vs 3
- Net communication value: comparison 1 vs 3

This mirrors the WO-07C ablation that separated bandwidth structure from field content in SPS.

## Files

- `business-questions.md` — customer segment, value proposition, validation plan, revenue model, constraints
