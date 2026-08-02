# Paper Outline: How Long Does a Rotating Field Have to Spin Before Team Communication Stops Helping?

**Target:** NeurIPS 2027 (9 pages + appendix)
**Fallback:** ICLR 2028

---

## Working title options

1. "Autocorrelation Time Determines the Value of Shared Memory in Non-Stationary Multi-Agent Collection"
2. "When Does Team Communication Degrade? Memory Limits Under Rotating Fields"
3. "L_critical ~ 1/ω: The Memory-Rotation Trade-off in Multi-Agent Coordination"

Option 3 is the most concrete and reviewer-memorable. Option 1 is most scientifically precise.

---

## Abstract (draft)

> We study multi-agent coordination under a time-varying latent field whose direction rotates at angular rate ω. Extending the confirmed SPS-C03 baseline (stationary field, +1.19 particles over independent agents), we ask: what is the minimum observation memory length L required to maintain a positive coordination benefit — and how does this critical length scale with ω? We show that L_critical(ω) ≈ c/ω across rotation speeds spanning an order of magnitude, where 1/ω is the field autocorrelation time. Observations older than 1/ω steps are approximately uncorrelated with the current field direction and contribute angular bias rather than SNR to the team estimate. We compare sliding-window and exponential-decay implementations and find [result pending experiments]. We map three real-world non-stationary collection domains to estimated ω values and compute their implied L_critical. The result provides a principled memory-window selection guideline: set L ≈ c/ω for your system's rotation rate.

---

## Section outline

### 1. Introduction (~1 page)

**Hook:** Communication protocols in deployed MARL systems are designed for one environment and run in another. Environments drift. A team that coordinates on a velocity summary trained in a stationary field will eventually coordinate on a stale signal if the field rotates.

**The question:** How fast can the field rotate before coordination becomes worse than independence — and what memory window mitigates this?

**Main result:** L_critical ~ 1/ω. The field autocorrelation time is the relevant timescale.

**Scope:** Extends SPS-C03 (stationary field, M=4) along the temporal axis. FR-B3 extends it along the (ρ,κ) spatial axes. Both papers characterise the operating regime of the shared_summary_v2 controller orthogonally.

---

### 2. Related work (~0.75 page)

**Non-stationary MARL:**
- Chang et al. (2023) on distribution shift in cooperative MARL — no communication memory analysis
- Online MARL with time-varying rewards — typically assumes slow drift; no explicit L_critical characterisation

**Memory in MARL:**
- R2D2 (Kapturowski et al. 2019): recurrent memory for single-agent; no principled memory-length prescription
- LSTM-MARL (Foerster et al. 2018): uses recurrent networks; no interpretable T_corr derivation
- *Gap:* No existing work derives a principled memory-window length from field autocorrelation time

**Communication under non-stationarity:**
- TarMAC (Das et al. 2019): targeted communication with attention; empirical, no T_corr analysis
- IC3Net (Singh et al. 2019): gated communication; no temporal decay analysis

**Discounted/windowed estimation:**
- Discounted UCB (Kocsis & Szepesvari 2006): exponential discount for non-stationary bandits — the same λ = exp(−1/L) decay applied to rewards rather than field estimates
- SWUCB (Garivier & Moulines 2011): sliding window for non-stationary bandits — our window controller is the cooperative analogue

**Position:** This paper is the first to derive a communication memory prescription from the field autocorrelation time in a multi-agent collection task.

---

### 3. Model and theory (~1.25 pages)

**3.1 The rotating field model**
- θ(t+dt) = θ(t) + ω·dt, θ(0) ~ Uniform[0, 2π)
- At ω=0: SPS-C03. At ω=π/67: half-rotation per episode
- Pre-generated θ(t) sequences for matched counterfactuals

**3.2 Memory models**
- Sliding window: last L steps, count-weighted (Proposition 2)
- Exponential decay: λ = exp(−1/L), count-weighted
- Both reduce to SPS-C03 controller at L=all, ω=0

**3.3 Derivation of L_critical**
- Team SNR under angular bias: SNR(L,ω) ~ √L · cos(ω·L·dt/2)
- Autocorrelation time T_corr = 1/ω
- L_critical prediction: c/ω
- Expected c from SPS parameters

---

### 4. Experimental setup (~0.75 page)

- Grid: 4 ω × 5 L × 2 methods = 40 conditions
- Matched seeds, shared θ(t) sequences, 8 seeds per cell
- Reproduction gate: (ω=0, Lall) must match SPS-C03
- Pre-registered analysis (cite `experiments/grid-design.md`)

---

### 5. Results (~3 pages)

**5.1 Reproduction gate and anchor**
- (ω=0, Lall) matches SPS-C03 Δ̄ = +1.19 ✓

**5.2 L_critical per rotation speed**
- Table: L_critical(ω) for sliding window and exponential decay
- Plot: sign count vs L, one panel per ω
- Main finding: L_critical ~ 1/ω confirmed / disconfirmed

**5.3 The 1/ω fit**
- Log-log plot of L_critical vs ω
- Fitted c, R², residuals
- Comparison to theoretical prediction

**5.4 Sliding window vs exponential decay**
- At each ω: which achieves lower L_critical?
- Does the winner change with ω?

**5.5 Team benefit degradation (Q3)**
- Δ̄(L, ω) for L ≥ L_critical
- Does Δ̄ decrease monotonically with ω?

---

### 6. Domain mapping (~0.5 page)

| Domain | Characteristic ω | Estimated L_critical | Operational feasibility |
|---|---|---|---|
| Tidal current (AUV) | π / (12hr / dt) | ~hours of memory | Feasible with onboard storage |
| Atmospheric wind rotation (UAV) | π / (24hr / dt) | ~half-day memory | Requires persistent state |
| Financial order flow (regime change) | Problem-dependent | Depends on regime duration | L must be tuned per regime |

---

### 7. Limitations (~0.5 page)

- Uniform spatial field: no spatial heterogeneity in rotation rate
- Scripted policies: learned policies may adapt L endogenously
- c is estimated from SPS parameters; different α requires re-estimation
- Monotone rotation: real fields have irregular non-stationarity, not constant ω

---

### 8. Conclusion (~0.25 page)

Set L ≈ c/ω. The field autocorrelation time is the right timescale for memory-window selection in non-stationary multi-agent collection. This paper identifies c empirically for the SPS regime; generalisation to other α, M, and domain parameters is future work.

---

## Appendix

- A: Full per-cell result tables (all 40 conditions)
- B: Reproduction gate verification
- C: Proof sketch for SNR(L,ω) derivation
- D: Alternative memory models (power-law decay)
- E: Replication instructions
- F: Connection to SPS-C03 and FR-B3

---

## Submission checklist

- [ ] ω=0 reproduction gate documented and passed
- [ ] Pre-registration matches analysis in `experiments/grid-design.md`
- [ ] All 40 conditions reported, including any L_critical = undefined cells
- [ ] 1/ω fit: R² reported, residuals discussed
- [ ] Kill criteria addressed explicitly
- [ ] Related work: R2D2, SWUCB, Discounted UCB, TarMAC cited
- [ ] Domain mapping: ω estimates sourced
- [ ] Code and configs for rotating-field environment in appendix
