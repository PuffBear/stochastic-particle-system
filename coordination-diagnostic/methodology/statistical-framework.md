# Statistical Framework for the Coordination Diagnostic

---

## Overview

The diagnostic produces two paired difference sequences — Delta_content(s) and Delta_structure(s) — across a seed set of size N. This document specifies how to summarise, interpret, and formally test those sequences.

---

## Descriptive summary (all engagement sizes)

For each estimand Delta ∈ {Delta_content, Delta_structure, Delta_net}:

| Statistic | Formula | Interpretation |
|---|---|---|
| Mean | mean(Delta) | Average per-episode effect |
| SD | std(Delta) | Episode-to-episode variability |
| Sign count | #{s : Delta(s) > 0} / N | Fraction of episodes where condition A > B (or B > C) |
| Min / Max | — | Range of individual episode effects |

**Reading the sign count:** A sign count of 5/8 means the effect is positive in most episodes but not all. A sign count of 8/8 is a strong positive signal. A sign count of 3/8 or lower indicates the communication is net-negative on content.

---

## Diagnostic gate (N=8, descriptive)

For a go/no-go engagement decision, use the sign count gate:

- **Content gate passes** if sign(Delta_content) ≥ 5/8
- **Structure gate passes** if sign(Delta_structure) ≥ 5/8

These thresholds are set at the 5% level of the sign test under H₀: P(Delta > 0) = 0.5. They are descriptive — the gate is not a formal hypothesis test and does not control error rates.

**Gate interpretation matrix:**

| Content gate | Structure gate | Diagnosis |
|---|---|---|
| Pass | Pass | Protocol adds value; content is informative beyond format |
| Fail | Pass | Format coordination; content is noise or harmful |
| Pass | Fail | Content helps but structural scaffold does not; unusual — check assumption 3 |
| Fail | Fail | Communication adds no value; investigate whether channel is active |

---

## Formal confirmatory test (N=32, pre-registered)

For a formal claim suitable for a V&V report or regulatory submission, use the pre-registered one-sided paired studentized bootstrap lower bound.

**Procedure:**
1. Compute paired differences Delta(s) = Y_s(A) - Y_s(C) for s = 1, ..., 32
2. Compute the sample mean mu_hat and standard error SE = std(Delta) / sqrt(N)
3. Draw B=9999 bootstrap samples of Delta with replacement; compute bootstrap mean mu_b* and bootstrap SE SE_b* for each
4. Compute studentized bootstrap statistics: T_b* = (mu_b* - mu_hat) / SE_b*
5. Estimate the 95th percentile q_0.95 of the T_b* distribution
6. One-sided 95% lower bound: L = mu_hat - q_0.95 * SE

**Claim threshold:** L > 0 establishes a positive net coordination value with 95% confidence.

**Minimum relevant effect:** Must be specified before running confirmatory seeds. The SPS reference is 1.0 unit on the team yield metric; translate to the client's metric before pre-registering.

---

## Power guidance

From SPS pilot data (SD ≈ 2.44 for Delta_net at alpha=0.06, M=4):

| Effect size (in metric units) | Required N for 80% power |
|---|---|
| 0.5 | ~130 seeds |
| 1.0 | ~40 seeds |
| 2.0 | ~12 seeds |

For most engagement contexts, 32 seeds will be sufficient for effects ≥1.0 metric unit. If the client's metric has higher variance than the SPS reference, run a 8-seed pilot first and estimate SD before committing to a seed count.

---

## Interpreting the content vs structure decomposition

The decomposition Delta_net = Delta_content + Delta_structure is additive by construction. Common patterns and their implications:

**Pattern 1: Delta_content ≈ Delta_net, Delta_structure ≈ 0**
Content drives all gain. The communication format is irrelevant — agents could receive information in any format and get the same benefit. This suggests the information itself is valuable and the channel design is not the bottleneck.

**Pattern 2: Delta_structure ≈ Delta_net, Delta_content ≈ 0**
Structure drives all gain. The protocol is providing coordination scaffold value — agents behave better when they expect to receive a signal, regardless of what it contains. This is the "bandwidth structure" finding from SPS WO-07C (+1.63 from structure, +1.0 from content). Implies the message content could be improved without changing the channel design.

**Pattern 3: Delta_content < 0, Delta_structure > Delta_net**
Content is actively harmful — scrambled messages outperform actual messages. This is the v1 failure mode from SPS WO-07: equal-weight averaging caused correlated team failures that random message assignment avoided. Diagnosis: the aggregation rule violates the sufficient statistic for the underlying quantity (see Proposition 2).

**Pattern 4: Both ≈ 0**
Communication is inert. Either the channel is not being used by the policy, or the task does not require coordination. Check that assumption 3 (no leakage) holds and that the policy actually conditions on incoming messages.

---

## Reporting format for a provenance certificate

The certificate should report, at minimum:

```
Coordination diagnostic summary
================================
Seeds: N=16 (diagnostic) / N=32 (confirmatory)
Condition A: [protocol name]
Condition B: permuted messages
Condition C: no messages

Gain from content (A vs B):
  Mean: +X.XX  SD: X.XX  Sign count: K/N
  Gate: PASS / FAIL (threshold K*/N)

Gain from structure (B vs C):
  Mean: +X.XX  SD: X.XX  Sign count: K/N
  Gate: PASS / FAIL

Net coordination value (A vs C):
  Mean: +X.XX  SD: X.XX
  [Confirmatory] 95% lower bound: +X.XX [ABOVE / BELOW zero]

Diagnosis: [one of the four pattern descriptions above]
Assumption verification: [PASSED / FLAGGED — see notes]
```
