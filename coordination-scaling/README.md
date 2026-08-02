# Coordination Scaling: Learned Channels and √M Empirical Validation

**Research ideas:** FR-B1 (learned channel vs. sufficient statistic) + FR-B2 (√M team-size scaling)
**Target venues:** ICLR 2027 · NeurIPS 2027
**Status:** 2–3 months out; MARL baselines (WO-08) provide context

## Why these two ideas belong together

FR-B1 asks whether MARL can discover the theoretically optimal communication structure from data. FR-B2 asks whether the predicted √M gain scaling holds empirically. Both questions test the same underlying theory from different angles:

- If the learned channel matches the sufficient statistic (FR-B1), the theory explains what MARL learns and why.
- If gain scales as √M (FR-B2), the theory correctly predicts how team benefit grows with team size.

A paper confirming both is a strong empirical validation of the theoretical framework. A paper where one or both deviate identifies exactly where the theory breaks down — equally publishable and arguably more interesting to ICLR reviewers.

## Paper framing

> "Do multi-agent systems learn what theory prescribes? Empirical tests of sufficient statistics and scaling laws in stochastic particle collection."

The interpretability angle (does learned = sufficient?) provides the ICLR hook. The scaling law (does gain ~ √M?) provides a clean empirical takeaway that survives even if the interpretability result is murky.

## Files

- `research-questions.md` — primary question, sub-questions, hypotheses, estimands, kill criteria
- `experimental-design.md` — training protocol, representation analysis, scaling sweep
