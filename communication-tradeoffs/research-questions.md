# Research Questions: Communication Failure Modes and Bandwidth Tradeoffs

## Primary research question

In a stochastic multi-agent particle collection task with a planted directional field, is there a signal-to-noise and team-size regime where inter-agent communication provably reduces team performance relative to independent operation — and above that regime, does coordination gain scale predictably with channel capacity?

## Sub-questions

### Q1 — The correlated-failure boundary

At what combination of field strength α and team size M does the fraction of seeds where `shared_summary_v2` yields fewer captures than `capacity_matched_independent` exceed 15%?

**Hypothesis:** Failure fraction exceeds 15% when per-step estimation error is high relative to the pooling benefit. Expected at (α=0.03, M≥4) and (α=0.06, M≥8) — regimes where spatial diversity loss outweighs SNR gain.

**Estimand:** P(Y_s(shared) < Y_s(independent)) across a grid of (α, M) from matched seeds.

**The two failure modes to distinguish:**
- Mode A (fixable): sharing a bad estimate when per-agent estimation error is high. WO-07B's count-weighted v2 fix addressed this.
- Mode B (structural): eliminating beneficial spatial diversity — all agents converge on the same wrong direction and fail together. The open question is whether Mode B persists in the v2 controller at M=8.

**What's already known:** At (α=0.06, M=4), v2 achieves mean +1.19 (lower bound +0.459). The equal-weight v1 failed (4/8 seeds positive). Neither an α sweep nor an M sweep has been run.

### Q2 — Bandwidth vs. coordination gain

At the confirmed SPS-C03 condition (α=0.06, M=4), does coordination gain increase smoothly or step-wise as channel capacity K increases from 1 number to full local state?

**Hypothesis:** There is a qualitative jump between K=2 (v_x, v_y) and K=3 (v_x, v_y, f_valid). The validity count allows agents to detect low-confidence estimates and down-weight them — a structural change in the message's meaning, not just more information.

**Estimand:** Mean contrast Δ̄(K) = E[Y(shared_K) − Y(independent)] for each of five frozen channels.

**Five channels:**

| K | Content |
|---|---|
| 1 | f_valid only |
| 1 | v_x only |
| 2 | v_x + v_y |
| 3 | v_x + v_y + f_valid (confirmed SPS-C03) |
| full | Complete local observation broadcast |

All channels use count-weighted team aggregation (Proposition 2 weighting) to isolate channel content from aggregation quality.

## Kill criteria

- **Q1 kill:** Failure fraction never exceeds 15% in any tested (α, M) cell — WO-07 was entirely attributable to the equal-weight bug. No new finding.
- **Q2 kill:** Gain monotonically increasing with overlapping confidence intervals everywhere — no structure, no design principle.
