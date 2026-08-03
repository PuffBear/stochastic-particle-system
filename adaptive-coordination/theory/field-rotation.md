# Theory: Rotating Field Model and L_critical Derivation

---

## Field model

The latent field direction evolves as:
```
θ(t + dt) = θ(t) + ω · dt,   θ(0) ~ Uniform[0, 2π)
```

where ω ≥ 0 is the rotation rate in radians per step (equivalently, radians / (dt seconds)).

At ω = 0: stationary field — exactly SPS-C03.
At ω = π/67: the field completes a half-rotation over one 67-step episode.

**Autocorrelation time:** The field direction at time t and time t + τ steps differs by ω·τ·dt radians (each step advances θ by ω·dt, not ω). The cosine similarity (relevant for velocity direction estimates) decays as cos(ω·τ·dt). The e-folding autocorrelation time is:
```
T_corr = 1 / (ω · dt)   (in steps)
```
Note the dt factor: it is the *per-step* rotation ω·dt, not the rotation rate ω, that determines how fast field estimates go stale. Observations more than T_corr steps old have cosine similarity < cos(1) ≈ 0.54 with the current field direction, and more than 2·T_corr steps old have cosine similarity < cos(2) ≈ −0.42 (potentially anti-informative).

---

## Team mean velocity estimate under memory length L

At time step t, agent i using a sliding window of the last L steps computes:
```
v̄_i(t, L) = Σ_{τ=max(0,t-L)}^{t} n_i(τ) · v̄_local_i(τ) / Σ n_i(τ)
```
where n_i(τ) is the number of particles observed by agent i at step τ and v̄_local_i(τ) is the local mean velocity at step τ.

The team mean (Proposition 2 sufficient statistic) is:
```
v̄_team(t, L) = Σ_i Σ_{τ} n_i(τ) · v̄_local_i(τ) / Σ_i Σ_{τ} n_i(τ)
```

**Signal component:** At step τ, the field direction is θ(τ) = θ(0) + ω·τ·dt. The expected velocity of a particle at step τ is α·(cos θ(τ), sin θ(τ)). The team mean at time t, using observations from steps [t−L, t], is biased toward the mean field direction over that window:
```
θ_mean(t, L) = θ(t) − ω·L·dt/2
```
(the midpoint of the window, in angle terms).

**Angular bias:** The team mean lags the true current field direction by ω·L·dt/2 radians. This bias grows with both ω and L.

**SNR tradeoff:** Longer L reduces estimation variance (more observations) but increases angular bias. The effective SNR of the team estimate, accounting for both:
```
SNR(L, ω, t) ≈ (α · √(M · K̄ · L · dt) / σ) · cos(ω · L · dt / 2)
```
where M·K̄ is the mean total observations per step across all agents, the first factor is the sensing gain from pooling, and the second is the cosine penalty from angular bias.

---

## Optimal memory length L*

Differentiating SNR(L, ω) with respect to L and setting to zero:
```
d/dL [√L · cos(ω·L·dt/2)] = 0
1/(2√L) · cos(ω·L·dt/2) − √L · (ω·dt/2) · sin(ω·L·dt/2) = 0
tan(ω·L·dt/2) = 1 / (ω·L·dt)
```

For small angles (ω·L·dt/2 << π): tan(x) ≈ x + x³/3, so 1/(ω·L·dt) ≈ ω·L·dt/2, giving:
```
L* ≈ √(2 / (ω·dt)²) = √2 / (ω·dt)
```

For ω = π/100 (mid-rotation speed) and dt = 0.02:
```
L* ≈ √2 / (π/100 · 0.02) = √2 / 0.000628 ≈ 2252 steps
```

This is much larger than our episode length (67 steps). **This is a consequence of the ω grid being in the wrong regime.** At ω=π/100 the per-step rotation ω·dt = 0.000628 rad ≈ 0.036°/step; over the full 67-step episode the field rotates only ~2.4°. For the rotation speeds originally tested ({π/200, π/100, π/50}), T_corr ∈ {3125, 1591, 795} steps — all far longer than the 67-step episode. In every tested episode the field is essentially stationary.

**Implication for the original experiments:** All 4 runs (9001–9032) at the old ω grid confirmed coordination benefit at all L and ω, but this is not genuine robustness to rotation — it is robustness to near-zero rotation. The old ω grid was off by a factor of ≈1/dt = 50.

**Corrected ω grid:** To put T_corr within the episode, we need ω·dt ≥ 1/67, i.e., ω ≥ 0.75 rad/step. The corrected grid targets T_corr ∈ {67, 33, 10, 3} steps:

| Label | ω (rad/step) | ω·dt (rad/step) | T_corr (steps) | Total rotation (67 steps) |
|---|---|---|---|---|
| very_slow | 0.75 | 0.015 | 67 | 1.0 rad (57°) |
| slow | 1.5 | 0.030 | 33 | 2.0 rad (115°) |
| mid | 5.0 | 0.10 | 10 | 6.7 rad (383°, >1 rotation) |
| fast | 17.0 | 0.34 | 3 | 22.4 rad (>3 rotations) |

For the rotation speeds we test, the SNR is monotone increasing in L within a single episode — the cosine penalty doesn't dominate at these ω values.

**Implication:** L* >> episode length for tested ω. The L_critical boundary exists not because the team estimate degrades at large L, but because the independent arm's estimate improves less with L. The coordination gain Δ̄(L, ω) = shared(L,ω) − independent(L,ω) has its own L_critical independent of the SNR maximum.

---

## Revised L_critical argument

The independent arm uses L steps of its own local observations. Its SNR scales as:
```
SNR_indep(L, ω) ≈ (α · √(K̄ · L · dt) / σ) · cos(ω·L·dt/2)
```

The shared arm benefits from M-fold more observations:
```
SNR_shared(L, ω) ≈ (α · √(M · K̄ · L · dt) / σ) · cos(ω·L·dt/2)
```

The ratio SNR_shared / SNR_indep = √M — independent of L and ω. **This means the coordination benefit does not degrade with L or ω as long as all agents use the same L.**

But this assumes agents are pooling correctly. The issue arises from angular heterogeneity: when the field rotates, agents at different locations may have observed different effective field directions (due to different particle neighbourhoods at each step). The team mean conflates observations from different field directions.

**The actual L_critical mechanism:** At high ω and large L, the variance of θ_effective — the field direction implied by the pooled observations — increases because older observations correspond to a different θ. The noise is not Gaussian around the current θ but is systematically biased by ω·L·dt/2 with additional spread from the agent-to-agent variation in observation timing. This spread grows with ω·L.

L_critical is where this systematic angular spread exceeds the sensing gain from pooling. A precise derivation requires knowing the distribution of per-agent observation counts and their correlation structure — this is addressed in the analysis after experiments.

**Corrected working prediction:** Using T_corr = 1/(ω·dt):
```
L_critical(ω) ≈ c / (ω · dt)   (in steps)
```
equivalently c' / (ω·dt) where c' = c·dt is a dimensionless constant. From SPS parameters, the angular noise per step is approximately σ/(α·√K̄) ≈ 0.06/(0.06·√8) ≈ 0.35 rad. The bias ω·L·dt/2 exceeds the noise when L ≈ 0.7/(ω·dt), suggesting c ≈ 0.7–1.0. Experiments will measure it directly.

---

## Numerical predictions (corrected ω grid)

Using c/(ω·dt) with c ≈ 1.0:

| ω (rad/step) | ω·dt (rad/step) | T_corr (steps) | Predicted L_critical |
|---|---|---|---|
| 0 | 0 | ∞ | all steps (SPS-C03) |
| 0.75 | 0.015 | 67 | ~67 steps |
| 1.5 | 0.030 | 33 | ~33 steps |
| 5.0 | 0.10 | 10 | ~10 steps |
| 17.0 | 0.34 | 3 | ~3 steps |

**Old grid (INCORRECT — in near-stationary regime):**

| ω (rad/step) | ω·dt (rad/step) | T_corr (steps) | Why it failed |
|---|---|---|---|
| π/200 ≈ 0.016 | 0.000314 | 3185 | T_corr >> episode length |
| π/100 ≈ 0.031 | 0.000628 | 1591 | T_corr >> episode length |
| π/50 ≈ 0.063 | 0.00126 | 794 | T_corr >> episode length |

These predictions are written before experiments. The corrected grid will test whether the empirical L_critical matches the predicted values and whether a single c fits all four ω levels.

---

## Exponential decay vs sliding window

For exponential decay with λ = exp(−1/L), the effective window is:
```
L_eff = 1 / (1 − λ) ≈ L   (for L >> 1)
```
but the weighting is smooth — observations at the cutoff are not abruptly zeroed. The angular bias of the exponential mean is:
```
θ_bias_decay ≈ ω · L · dt   (one effective half-window)
```
compared to ω · L · dt / 2 for the sliding window (midpoint of window).

**Prediction:** At the same nominal L, exponential decay has approximately twice the angular bias of the sliding window. This should make exponential decay perform *worse* at the same L — the opposite of the smoothness intuition.

However, at L = L_critical, the exponential decay places more weight on very recent observations and less on observations near the L-step boundary. In practice, the effective window seen by the exponential decay is concentrated closer to the current step than the sliding window at the same L. This may offset the bias difference.

This is an open empirical question. The prediction is written down; the experiment resolves it.
