# Communication Failure Modes and Bandwidth Tradeoffs

**Research ideas:** FR-A1 (correlated-failure boundary) + FR-A2 (bandwidth vs. gain curve)
**Target venues:** AAMAS 2027 · ICML 2027
**Status:** Ready to develop once MARL baselines (WO-08) complete

## Why these two ideas belong together

FR-A1 establishes the failure regime: the (SNR, team-size) region where sharing a team signal provably hurts relative to independent operation. FR-A2 characterizes what happens above that threshold: how coordination gain scales with channel capacity from 1 number to full state.

Together they form a complete, unified paper — a systematic map of communication value in stochastic multi-agent systems, from the failure mode at low signal quality to the saturation curve at high bandwidth. The SPS-C03 confirmation (+1.19 particles) and the WO-07 failure story (equal-weight sharing hurt on 4/8 seeds) give the paper a concrete, reproducible case study anchoring both claims.

## Two-claim structure

**Claim 1 (FR-A1 — the warning):** There exists a (SNR, team-size) boundary below which sharing synchronized decisions hurts more than isolated failures — and this boundary is predictable from per-step estimation noise.

**Claim 2 (FR-A2 — the design principle):** Above that boundary, coordination gain scales with channel capacity in a way that is either smooth (monotone, predictable) or step-wise (identifying the minimum viable communication structure).

The combination is stronger than either alone: a reviewer can object to either claim in isolation, but the two together define an operating envelope with both a floor (failure regime) and a ceiling (saturation curve).

## Files

- `research-questions.md` — primary question, sub-questions, hypotheses, estimands, kill criteria
- `experimental-design.md` — conditions, seeds, analysis plan
