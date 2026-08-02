# Future Research Ideas

**Status as of 2026-08-01.** SPS-C03 confirmed coordination. The research ideas below are assessed against that confirmed baseline. Ideas are grouped by the conference they most naturally target, not by internal work-order numbering. Each idea has a one-line pitch, the open question, why it is non-trivial (the *hard part*), and a kill criterion.

Inactive directions only. None may consume active-paper resources without a Program Director decision. All hypotheses are unvalidated.

---

## Tier 1 — Ready to develop now (blocked only by MARL baselines)

These build directly on the confirmed C03 result and the WO-07 failure story.

---

### FR-A1 — When does sharing hurt? The correlated-failure boundary
**Target venue:** AAMAS 2027 / ICLR 2027 (multi-agent learning track)

**One-line pitch:** SPS-WO-07 showed that an equal-weight shared signal made performance *worse* than doing nothing on 4/8 seeds; WO-07B fixed it. The boundary between "sharing helps" and "sharing synchronizes failure" is unexplored.

**Open question:** At fixed communication capacity (3 numbers), is there a signal-to-noise ratio or team-size regime where sharing provably hurts relative to independent operation — and can that boundary be predicted from the channel's estimation noise alone?

**Hard part:** Distinguishing two failure modes that look identical from the outside: (a) sharing a bad estimate (fixable by better aggregation) vs. (b) eliminating beneficial spatial diversity (not fixable without adding stochasticity). A clean experimental design has to manipulate estimate quality and spatial dispersion independently.

**Why this is AAMAS/ICLR-level:** The correlated-failure story from WO-07 is a concrete, reproducible demonstration of a phenomenon that appears in theory but is rarely shown with a controlled benchmark. It directly addresses the "when to communicate" question from a risk perspective rather than a reward perspective.

**Minimum experiment:** Sweep `alpha` (SNR) and `M` (team size) over a pre-registered grid; compute the fraction of seeds where shared < independent, and the fraction where shared < stationary. Regress against estimated per-step field estimation error.

**Kill criterion:** The failure fraction is never >15% in any tested regime, meaning WO-07 was an outlier attributable to the equal-weight bug alone, not a fundamental phenomenon.

---

### FR-A2 — Communication bandwidth vs. coordination gain
**Target venue:** AAMAS 2027 / ICML 2027

**One-line pitch:** The confirmed channel is exactly 3 numbers. Is there a clean diminishing-returns curve as the channel widens from 1 number to full state?

**Open question:** Does coordination gain scale with channel capacity according to a predictable information-theoretic curve, or are there qualitative jumps at specific capacity thresholds (e.g., adding a validity count after just velocity)?

**Hard part:** Designing channel variants that are capacity-matched in a meaningful way. Adding one number changes both information content and the agent's ability to detect bad estimates. These two effects need to be separated.

**Why this is ICML-level:** If the gain-vs-capacity curve is smooth and predictable, it provides a practical design principle. If it has jumps, it identifies a minimum viable communication structure — which is a much more interesting theoretical result. Either outcome is publishable.

**Minimum experiment:** Five frozen channels: (1) f_valid only, (2) v_x only, (3) v_x + v_y, (4) v_x + v_y + f_valid [current], (5) full local observation broadcast. Run all five at alpha=0.06 on 16 seeds with matched capacity budget.

**Kill criterion:** Gain is monotonically increasing with no structure and confidence intervals overlap everywhere — curve has no informative shape.

---

### FR-A3 — Ablation completeness: message-shuffle and leave-one-out
**Target venue:** AAMAS 2027 (required for submission, not optional)

**Note:** FR-A3 belongs to the current SPS paper submission. It is listed here for completeness but should not be treated as a future paper direction.

**One-line pitch:** The current C03 result is a contrast, not a causal attribution. AAMAS reviewers will demand message-shuffled and leave-one-agent-out controls before accepting the communication claim.

**Kill criterion:** Shuffled message achieves the same or better gain as the real message — the content carries no information beyond the structural format.

---

## Tier 2 — Strong research directions (2–3 months out)

---

### FR-B1 — Learning the optimal bounded channel
**Target venue:** ICLR 2027 / NeurIPS 2027

**One-line pitch:** We hand-designed a 3-number sufficient statistic using a Fisher-Neyman factorization argument. Can a learned encoder discover the same structure from data — or find something better?

**Open question:** Given a fixed channel capacity of K numbers (K=3 as the baseline), does end-to-end learned communication recover the count-weighted mean velocity + validity fraction structure, or does it find a qualitatively different representation?

**Hard part:** Making the comparison fair. A learned channel has access to the full local observation and can encode task-specific structure; the hand-designed channel made explicit simplifying assumptions. You need a way to evaluate whether learned representations are *interpretably equivalent* to the sufficient statistic, not just whether they achieve higher reward.

**Why this is ICLR-level:** The interpretability angle — does MARL recover a sufficient statistic when one exists? — is a rare case where there is a known theoretical answer to compare against. Most interpretability papers study systems where the ground truth is unknown.

**Minimum experiment:** Train CommNet (or DIAL) with a bottleneck of 3 values; decode the learned channel and compare to hand-designed channel on held-out seeds. Measure whether the learned representations correlate with (v_x, v_y, f_valid).

**Kill criterion:** Learned channel achieves same gain as hand-designed on held-out seeds but the representations are uninterpretable — no clear comparison possible.

---

### FR-B2 — Team size scaling: does √M amplification hold empirically?
**Target venue:** ICML 2027 / AAMAS 2027

**One-line pitch:** Theory predicts that shared SNR = ρ·√(MKf) while independent per-agent SNR = ρ·√(Kf), so sharing should gain √M over independent. At M=4, that is a factor of 2. The observed C03 gain (+1.19 captures) is smaller. Does the scaling hold across M?

**Open question:** Does the coordination gain scale as √M across team sizes, or is there a maximum team size beyond which spatial diversity loss from sharing outweighs the SNR gain?

**Hard part:** At larger M, the correlated-failure problem from WO-07 becomes worse (all M agents fail together), but the estimation noise is lower. These effects pull in opposite directions. The v2 field+density blend may not scale well at M=8 or M=16 without retuning blend_w.

**Minimum experiment:** M ∈ {1, 2, 4, 8} with matched total sensing budget (K·M constant). Run 16 seeds per M level. Plot mean contrast vs. √M.

**Kill criterion:** Gain is flat or decreasing across M — no scaling relationship.

---

### FR-B3 — Catchability-aware scale transfer (nondimensional benchmark axis)
**Target venue:** ICML 2027 (benchmark track)

**One-line pitch:** Our current benchmark has two confounded axes: sensing difficulty (ρ = α√dt/σ) and control authority (κ = α/v_max). Separating them into a two-axis benchmark space would make results transferable across physical domains.

**Minimum experiment:** Three scale-equivalent configs at κ ∈ {0.25, 0.5, 1.0}; 8 seeds each; plot coordination gain curves against κ.

**Kill criterion:** Curves don't collapse across κ values at fixed ρ, meaning the factorization is wrong or there are other confounders.

---

### FR-B4 — Nonstationary fields: online field estimation with bounded memory
**Target venue:** IJCAI 2027 / ICLR 2027

**One-line pitch:** The current task has a fixed field direction per episode. If the field rotates or shifts during an episode, agents must trade off tracking accuracy against capture effort. How much memory is needed to maintain coordination benefit?

**Minimum experiment:** Three field rotation speeds; 3-number memory-augmented channel; measure the step at which coordination gain drops below zero.

**Kill criterion:** Any nonzero memory maintains coordination benefit at all tested rotation speeds — no informative memory-length boundary.

---

## Tier 3 — Long-horizon ideas (6+ months out)

### FR-C1 — Offline multi-agent evaluation from paired trajectory datasets
**Target venue:** NeurIPS 2027 (datasets and benchmarks track)

**Key dependency:** Needs a confirmed coordination effect (done) and a dataset of sufficient coverage (not yet built).

### FR-C2 — Heterogeneous teams: sensing-action specialization near weak signals
**Target venue:** AAMAS 2027

**Key dependency:** Needs a clean homogeneous baseline (done) and a theoretical extension of Proposition 2 to heterogeneous agents.

### FR-C3 — Strategic particles: pursuit-evasion with stochastic structure
**Target venue:** AAMAS 2027 (game-theoretic track)

**Key dependency:** Requires passive-particle results to be solid (done) and a clear definition of the evader's information and action budget.

---

## Archive (superseded or killed)

- **FR-001**: Retired into FR-A3 (ablation completeness).
- **FR-002**: Event-keyed tie invariance — engineering validation, not a paper direction.
- **FR-003**: Growing-geometry aggregation — inactive until fixed-geometry results are published.
- **FR-007**: Offline MARL from paired trajectories — promoted to FR-C1.
- **FR-012**: Simultaneous inference calibration — methodology supplement, not a standalone paper.
- **FR-013**: Timestep convergence audit — completed as SPS-WO-06, confirmed adequate.
