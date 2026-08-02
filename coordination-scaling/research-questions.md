# Research Questions: Coordination Scaling

## Primary research question

Given a fixed 3-slot communication channel, does end-to-end MARL training recover the count-weighted mean velocity and validity fraction structure predicted by Proposition 2 — and does team coordination gain scale empirically as √M across team sizes M ∈ {1, 2, 4, 8} at matched total sensing budget?

## Sub-questions

### Q1 — Does MARL recover the sufficient statistic?

**Setup:** Train CommNet with a 3-value communication bottleneck on SPS at α=0.06, M=4. Decode learned representations and compare to (v_x, v_y, f_valid).

**Hypothesis:** The learned representations correlate with (v_x, v_y) but not cleanly with f_valid — the validity fraction requires a discrete threshold operation that soft encoders resist. The learned channel will match the hand-designed channel in aggregate reward but via a different internal representation.

**Estimand:** Spearman correlation between each bottleneck dimension and (v_x, v_y, f_valid, n_valid, local_density) across 500 held-out trajectories. Secondary: coordination gain comparison on 16 held-out matched seeds.

**Architectures:** CommNet (soft mean, 3-slot bottleneck), DIAL (differentiable discrete, 3-bit), hand-designed SPS-C03 baseline.

**Key distinction:** Reward-equivalent but representationally different = MARL found an alternative sufficient statistic (positive result). Reward-inferior with uninterpretable representations = MARL failed to find the optimal structure (publishable negative result).

### Q2 — Does coordination gain scale as √M?

**Theoretical prediction:** Shared SNR = ρ·√(M·Kf) vs. independent ρ·√(Kf), predicting a √M gain factor. At M=4, the predicted amplification is 2×; the observed C03 gain corresponds to ~1.14× — theory overpredicts.

**Hypothesis:** √M scaling holds at M ∈ {2, 4} but breaks at M=8 because spatial diversity loss grows with M and blend_w (tuned at M=4) is not optimal at M=8.

**Estimand:** Mean contrast Δ̄(M) = E[Y(shared) − Y(independent)] at matched total sensing budget (K·M = constant = 8). Fit Δ̄(M) ~ M^β; test H₀: β = 0.5.

**Matched budgets:**

| M | K per agent | Total obs |
|---|---|---|
| 1 | 8 | 8 — no sharing possible |
| 2 | 4 | 8 |
| 4 | 2 | 8 — SPS-C03 condition |
| 8 | 1 | 8 |

## Kill criteria

- **Q1 kill:** Same reward as hand-designed but bottleneck activations are uninterpretable across 500 held-out trajectories. No representational comparison possible.
- **Q2 kill:** Gain flat or decreasing across M at matched sensing budget. No power-law relationship holds at any tested scale.
