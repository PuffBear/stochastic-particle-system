# FR-B4 Adaptive Coordination — Full Publication Plan

**Target venue:** ICLR 2028 (primary) · NeurIPS 2028 (fallback)
**Submission deadline:** ICLR 2028 ~October 2027
**Realistic timeline from now:** 10–12 months to ICLR submission

---

## The paper in one paragraph

We extend the SPS-C03 multi-agent collection task to a time-varying field and ask: how short can an agent's memory window be before the coordination benefit disappears? We show empirically that the critical memory length L_critical scales as the field autocorrelation time 1/ω, where ω is the rotation speed of the latent field direction. This follows from an information-theoretic argument: observations older than 1/ω steps are approximately uncorrelated with the current field direction and contribute noise rather than signal to the team mean estimate. We test this prediction across four rotation speeds and five memory lengths, compare sliding-window and exponential-decay implementations, and show that team benefit degrades with ω even above L_critical due to increased inter-agent correlation. The result generalises the SPS-C03 finding from stationary to non-stationary fields and provides a principled guideline for memory window selection in deployed systems.

---

## Why NeurIPS 2027

NeurIPS rewards papers that combine a clean theoretical prediction with rigorous empirical validation and a connection to deployed systems. The L_critical ~ 1/ω prediction is exactly this structure: falsifiable, grounded in information theory, tested across multiple ω levels.

ICLR 2027 is ruled out by timeline: the prerequisite engineering alone takes 4–6 weeks, leaving insufficient time before the ~October 2026 deadline. ICLR 2028 (deadline ~October 2027) is the fallback if NeurIPS 2027 is missed.

**Why not AAMAS:** AAMAS suits mechanism papers (how coordination works); this paper is about temporal limits (when it stops working). The theoretical grounding and clean prediction fit NeurIPS better.

**Why not ICML:** ICML is the right home for FR-B3 (parameterisation). FR-B4's contribution is specifically about non-stationarity and memory — closer to NeurIPS's ML systems / theory flavour.

---

## Phase 0: What we already have

| Asset | Status |
|---|---|
| SPS-C03 confirmed result (ω=0, L=all) | ✅ Anchor — +1.19, lower bound +0.459 |
| shared_summary_v2 controller (full history) | ✅ Implemented |
| capacity_matched_independent controller | ✅ Implemented |
| Matched counterfactual infrastructure | ✅ Pre-generated noise tensors |
| SPS environment (static field) | ✅ Confirmed valid |

What does NOT exist:
- Rotating field implementation
- Pre-generated θ(t) sequences for matched counterfactuals
- Sliding window and exponential decay controller variants
- ω=0 reproduction gate for the modified environment

---

## Phase 1: Theory — derive L_critical and verify the 1/ω prediction (Weeks 1–2)

**Deliverable:** `theory/field-rotation.md` — derivation of L_critical from the field autocorrelation time, numerical predictions for each tested ω, and the constant c.

**Key derivation:**
The team mean velocity estimate at time t using the last L steps has effective SNR:
```
SNR(L, ω) = (α · √(L · dt) / σ) · cos(ω · L · dt / 2)
```
The first term is the sensing gain from pooling L observations; the second is the cosine decay from the field having rotated ω·L·dt/2 radians since the midpoint of the window. SNR is maximised at L* = π / (ω·dt). Beyond L*, the cosine penalty dominates.

L_critical is smaller than L* because we need positive Δ̄ (coordination beats independent), not just maximum team SNR. The gap depends on the independent arm's SNR, which must be derived.

**Theoretical predictions to write down before experiments:**
- c (the constant in L_critical ≈ c/ω) estimated from SPS parameters
- Predicted L_critical at ω ∈ {π/200, π/100, π/50}
- Predicted ordering: sliding window vs exponential decay at high ω

**Kill criterion at this phase:** If the SNR formula does not have a well-defined maximum in L (e.g. monotone increasing for all tested L), the experiment will not find a boundary. Check this analytically before running.

---

## Phase 2: Prerequisite engineering (Weeks 2–5, parallel with Phase 1)

This is the main bottleneck. All four sub-tasks must be complete before any experiment runs.

### 2a: Rotating field environment

Modify the SPS environment to support θ(t+dt) = θ(t) + ω·dt with:
- ω as a config parameter (default 0 = SPS-C03)
- Pre-generation of θ(t) sequences as part of the seed infrastructure (so signal and null arms share the same rotation trajectory)
- θ(t) sequence stored in the episode manifest for reproducibility

**Validation:** At ω=0, the modified environment must produce identical results to the original SPS environment for the same seeds (byte-level match on episode summaries).

### 2b: Memory controller variants

Implement two new controller variants extending shared_summary_v2:

**shared_summary_window_L:** Uses only the last L timesteps of particle velocity observations. The count-weighted mean is computed over this window only.
```
v̄_team(t) = Σ_{τ=max(0,t-L)}^{t} n_i(τ) · v̄_i(τ) / Σ n_i(τ)
```

**shared_summary_decay_L:** Uses exponentially decaying weights with λ = exp(−1/L):
```
v̄_team(t) = Σ_{τ=0}^{t} λ^(t-τ) · n_i(τ) · v̄_i(τ) / Σ λ^(t-τ) · n_i(τ)
```

Both must reduce exactly to shared_summary_v2 at L = all steps and ω = 0.

### 2c: Baseline adaptation

The capacity_matched_independent controller must use the same memory model as the shared arm in each condition. If the shared arm uses a window of L, the independent arm must also use a window of L applied to its own local observations only. This ensures the only causal difference remains the message channel, not the memory structure.

### 2d: ω=0 reproduction gate

Run seeds 6001–6032 (32 SPS-C03 confirmed seeds) with the modified environment
at ω=0, L=1. Require Δ̄ ∈ [+0.69, +1.69] and sign count ≥20/32.

L=1 is used (not L=all) because the FR-B4 controller is not stateless — at
L=all it accumulates temporal observations which changes both arms' behaviour
relative to the stateless SPS-C03 policy. L=1 uses only the current step and
directly reproduces SPS-C03's per-step computation for the shared arm.

The FR-B4 independent arm applies density blend (same as the shared arm), so Δ
at L=1 will be slightly lower than SPS-C03's +1.19. The gate range accommodates
this. Gate must pass before Phase 3 begins.

---

## Phase 3: Run the experiments (Weeks 5–8)

**Grid:** 4 ω × 5 L × 2 methods = 40 conditions. 8 seeds per cell. Total: 640 episodes per arm = 1280 runs.

**Runtime estimate:** ~10s per episode. 1280 episodes ≈ 3.6 hours single-threaded; ~30 minutes on 8 cores.

**Seed plan:**
- Seeds 9001–9008: all non-zero ω conditions
- Seeds 1001–1008 (reused): ω=0 anchor

**Run order:** Start with ω=π/50 (fastest rotation) at L∈{1,3,10} — these cells are most likely to show the boundary and serve as an early sanity check. If no boundary appears at ω=π/50, revisit the theory before running the full grid.

**Quality checks:**
- At each ω, L=all (full history) must be worse than the SPS-C03 anchor — rotating field should degrade performance
- At each ω, L=1 should show degraded or near-zero Δ̄ (single-step memory is too noisy)
- Oracle baseline must remain highest across all conditions

---

## Phase 4: Analysis (Weeks 8–10)

**Primary:** For each (ω, method), find L_critical = smallest L with ≥60% positive seeds. Plot L_critical(ω) on log-log axes; test 1/ω fit.

**Secondary:** Compare sliding window vs exponential decay: does exponential decay achieve lower L_critical at high ω?

**Q3 analysis:** For L > L_critical at each ω, plot Δ̄ as a function of ω. Test whether Δ̄ decreases monotonically.

**Q4:** Fit L_critical = c/ω to three non-zero ω levels. Report c and R². If 1/ω fit has R² < 0.80, test alternative scalings.

---

## Phase 5: Domain mapping (Week 10)

**Deliverable:** `theory/domain-mapping.md` — real systems with time-varying drift, estimated ω values, implied L_critical.

**Target domains:**
- Tidal current monitoring (ω tied to tidal period — predictable)
- Atmospheric boundary layer (diurnal wind rotation — daily cycle)
- Financial order flow (intraday directional drift rotation — regime changes)

For each: estimate ω in rad/step (given dt=0.02s equivalent), compute predicted L_critical = c/ω, discuss whether the implied memory window is operationally feasible.

---

## Phase 6: Writing (Weeks 10–16)

**Target length:** 9 pages + appendix (NeurIPS format)

**Section outline:**

1. **Introduction** (~1 page): The non-stationarity problem in MARL communication. SPS-C03 as the stationary baseline. Main result: L_critical ~ 1/ω.

2. **Related work** (~0.75 page): Recurrent and memory-augmented MARL (R2D2, LSTM-MARL); communication under noise (TarMAC, IC3Net); online learning in non-stationary environments (discounted UCB, exp decay).

3. **The rotating field model** (~1 page): θ(t) dynamics; matched counterfactual design with pre-generated θ(t); memory model definitions; derivation of L_critical.

4. **Experimental setup** (~0.75 page): Grid design; ω=0 reproduction gate; seed plan.

5. **Results** (~2.5 pages): L_critical per ω (main result); 1/ω fit; sliding window vs exponential decay; Q3 team benefit under drift; Q4 theoretical check.

6. **Domain mapping** (~0.75 page): Real non-stationary systems with estimated ω and L_critical.

7. **Limitations** (~0.5 page): Uniform spatial field (FR-B3 spatial structure not combined here); scripted policies; the c constant needs re-estimation for different α.

8. **Conclusion** (~0.25 page).

---

## Kill criteria

**Primary kill:** No L_critical boundary found — Δ̄ > 0 even at L=1 for all tested ω. Either the theoretical SNR formula is wrong or the task is so forgiving that any memory length works.

**Scaling kill:** L_critical(ω) does not fit 1/ω and no alternative scaling emerges.

**Reproduction kill (Phase 2d gate):** Modified environment does not reproduce SPS-C03 at ω=0. Do not proceed to Phase 3.

**Timeline kill:** If Phase 2 engineering takes more than 8 weeks, NeurIPS 2027 deadline is at risk. In that case, pivot to ICLR 2028.
