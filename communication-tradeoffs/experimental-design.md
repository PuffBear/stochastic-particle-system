# Experimental Design: Communication Failure Modes and Bandwidth Tradeoffs

## Experiment 1 — Correlated-failure boundary sweep

| Factor | Levels |
|---|---|
| α (field strength) | 0.03, 0.06, 0.10, 0.15 |
| M (team size) | 2, 4, 8 |
| Seeds per cell | 16 matched pairs |
| Arms | shared_summary_v2 vs. capacity_matched_independent |

Fixed: N=256, dt=0.02, evaluation_steps=67, σ=0.06, arena frozen per SPS PROJECT.md.

**Primary outcome:** Failure fraction F(α, M) = P(Y_s(shared) < Y_s(independent)).

**Analysis:**
1. Report F(α, M) as a 4×3 heatmap with 95% Wilson CIs.
2. Fit logistic regression: logit(F) ~ α + log(M) + α·log(M). Report the estimated boundary where F̂ = 0.15.
3. Correlate F with estimated per-step field estimation error σ_est = σ / (α · √(M · Kf)).
4. Classify failing seeds by Mode A vs. Mode B using trajectory logs (f_valid level and agent direction agreement).

**Pre-registration gate:** At least one (α, M) cell must show F > 0.15 for the boundary claim to survive.

---

## Experiment 2 — Bandwidth vs. gain curve

| Channel ID | Content | Slots |
|---|---|---|
| single_valid | f_valid only | 1 |
| single_vx | v_x only | 1 |
| dual_velocity | v_x + v_y | 2 |
| triple_summary | v_x + v_y + f_valid (SPS-C03) | 3 |
| full_local | Complete local observation | ~16 |

Fixed: α=0.06, M=4, N=256, dt=0.02, 67 steps. 16 matched seeds per channel.

**Analysis:**
1. Plot Δ̄(K) vs. K with 95% bootstrap CIs.
2. Test for a jump between dual_velocity and triple_summary — is the CI gap larger than the gap from single_vx to dual_velocity?
3. Compute efficiency ratio: gain per additional slot at each transition.
4. Report triple_summary vs. full_local: gain remaining above the sufficient statistic.

**Pre-registration gate:** At least one pair of adjacent channels must show non-overlapping 95% CI.

---

## Shared infrastructure

- Pre-register 16 seed pairs before running either experiment.
- Matched pairs share: initial positions, Brownian noise tensor, field direction, tie-breaking randomness. The only causal difference is the communication condition.
- All arms run from a frozen environment commit; no code changes between conditions.
