# Paper Outline: When Does Team Communication Help? A Nondimensional Benchmark for Multi-Agent Collection

**Target:** ICML 2027 (8 pages + appendix)
**Fallback:** NeurIPS 2027 Datasets & Benchmarks (9 pages + unlimited appendix)

---

## Working title options

1. "When Does Team Communication Help? A Nondimensional Benchmark for Multi-Agent Collection Tasks"
2. "Sensing Difficulty and Control Authority Determine the Value of Multi-Agent Communication"
3. "A Two-Parameter Characterisation of Coordination Gain in Stochastic Collection Tasks"

Option 1 is most accessible. Option 2 is most precise. Option 3 is most neutral. Decide at writing phase.

---

## Abstract (draft)

> We introduce a two-parameter characterisation of multi-agent collection tasks: ρ, the per-observation signal-to-noise ratio of the field direction estimate, and κ, the ratio of signal drift speed to collector maximum speed. We show empirically that the coordination gain from a minimal shared communication channel — the mean team velocity direction and observation validity fraction — is approximately multiplicatively separable in ρ and κ across a 3×3 parameter grid. At the confirmed operating point (ρ=0.14, κ=0.20), shared communication produces a mean +1.19 additional unique particles captured over an identical-capacity independent controller (95% lower bound +0.46, N=32 matched seeds). We characterise the coordination regime — the region of (ρ, κ) space where communication reliably adds value — and provide a mapping from four real multi-agent collection domains (ocean AUV, agricultural UAV, search-and-rescue, oil spill) to the benchmark grid. The result gives practitioners a diagnostic: compute your system's (ρ, κ) and determine whether a low-bandwidth shared channel is likely worth implementing.

---

## Section-by-section outline

### 1. Introduction (~1 page)

**Opening hook:** Multi-agent communication is standard in deployed swarm systems, but whether it helps in any given application is rarely tested systematically. Engineering teams add communication because they expect it to help; they rarely run the controlled comparison to verify.

**The problem:** Existing multi-agent benchmarks (SMAC, MPE, Cooperative Navigation, MAMuJoCo) do not parameterise the task in terms of quantities that predict when communication helps. They report performance under fixed conditions; they do not characterise the operating regime.

**The contribution:**
1. A two-parameter task characterisation (ρ, κ) derived from first principles
2. Empirical evidence that coordination gain is approximately separable in ρ and κ
3. A 3×3 benchmark grid with reproducible results at each cell, anchored to a pre-registered confirmed result
4. A domain mapping showing where real systems fall in the grid

**Scope:** Scripted (not learned) policies; the shared channel is hand-designed. The question is whether any fixed low-bandwidth summary helps at all, not whether learned communication can do better. FR-B1/B2 addresses learned policies.

---

### 2. Related work (~0.75 page)

**Multi-agent coordination benchmarks:**
- SMAC (Samvelyan et al. 2019): StarCraft II micromanagement. Rich environment, no parameterisation of when communication helps.
- MPE / Cooperative Navigation (Lowe et al. 2017): Simple 2D, but tasks are not parameterised by difficulty dimensions that predict communication value.
- MAMuJoCo (Peng et al. 2021): Continuous control, no communication channel characterisation.
- *Gap:* None characterise the task by a parameter that predicts communication value.

**Communication in MARL:**
- CommNet (Sukhbaatar et al. 2016): Continuous communication; shows improvement but not when/why.
- DIAL (Foerster et al. 2016): Differentiable communication; task-specific results.
- TarMAC (Das et al. 2019): Targeted communication; empirical, no theory of when it helps.
- *Gap:* Literature shows communication can help; does not characterise the regime.

**Mobile particle collection:**
- Wang et al. (2025): Single mobile collector, no multi-agent component. We cannot claim first mobile collector.
- Löffler et al. (2023): Locally perceiving active particles, RL-trained. No κ-ρ parameterisation.

**Dimensionless parameterisation in related fields:**
- Péclet number (Pe): advection/diffusion ratio in fluid mechanics — structural analogue of κ
- Beverton-Holt catchability q ∝ 1/κ in fisheries ecology
- Damköhler number in reactive flows — another dimensionless ratio governing regime transitions

**Position:** This paper is the first to apply dimensionless parameterisation to the question of multi-agent communication value in collection tasks.

---

### 3. The ρ-κ parameterisation (~1 page)

**3.1 Task description**
- N=256 particles undergoing drift-diffusion; M=4 mobile collectors
- Collectors observe local particles; field direction θ unknown but fixed
- Metric: unique particles captured in 67-step window

**3.2 Derivation of ρ**
- Per-observation SNR of field direction estimate
- ρ = α·√dt / σ
- Low ρ: individual field estimates are noisy → shared team summary has highest value

**3.3 Derivation of κ**
- κ = α / v_max (drift speed / collector speed)
- κ < 1: collectors can outrun particles; κ > 1: cannot
- Connection to Péclet number, Beverton-Holt catchability

**3.4 The separability hypothesis**
- Δ̄(ρ, κ) ≈ C · g(ρ) · h(κ)
- Mechanistic justification: sensing gain (depends on ρ) and action gain (depends on κ) are approximately independent
- Predicted qualitative shape: g decreasing in ρ; h non-monotone, peak near κ ≈ 0.5

---

### 4. The SPS benchmark (~1 page)

**4.1 Environment**
- Arena, dynamics, sensing, action bounds (cite SPS paper / upstream work)
- Collector: shared_summary_v2 vs capacity_matched_independent
- Three-number shared channel: (mean v_x, mean v_y, f_valid) — count-weighted (Proposition 2)

**4.2 Experimental protocol**
- Matched counterfactual design: same Brownian noise tensor, initial positions, tie-breaking
- Pre-registered grid design and analysis (cite pre-registration document)
- 8 diagnostic seeds per cell; 16 confirmatory if gate passes
- Cell (mid, mid) anchored to SPS-C03: confirmed positive result

**4.3 The 3×3 grid**
- Table of (α, v_max) per cell
- ρ and κ values per cell

---

### 5. Results (~2 pages)

**5.1 Per-cell results**
- Table: mean Δ̄, SD, sign count, gate result for all 9 cells
- Cell (mid, mid): Δ̄ = +1.19 (confirmed, N=32)

**5.2 Separability test**
- Fitted multiplicative surface
- R² of fit
- 3×3 heatmap: observed vs fitted side-by-side

**5.3 The coordination regime**
- Contour plot of predicted Δ̄(ρ, κ) from fitted model
- Highlight the regime where Δ̄ > 1.0 and sign count ≥ 6/8

**5.4 Oracle gap**
- Oracle − shared_v2 per cell
- Shows remaining headroom; motivates FR-B1/B2 (learned policies)

---

### 6. Domain mapping (~0.75 page)

- Table: real systems, estimated (ρ, κ), predicted regime
- Agricultural UAV: confirmed beneficial regime
- Search-and-rescue ground robot: predicted zero-benefit (κ > 1)
- Ocean AUV: predicted low benefit (very low κ, high ρ — catching trivial, sensing easy)
- Practical guidance: how to estimate ρ and κ for your system

---

### 7. Limitations (~0.5 page)

- Uniform, static field (FR-B4 addresses time-varying)
- Scripted policies (FR-B1/B2 addresses learned)
- 2D symmetric arena (boundary effects minimal but untested in complex geometry)
- 3×3 grid is coarse (8 cells outside the confirmed point; finer grid is future work)
- Domain mapping estimates are rough; require domain calibration to be precise

---

### 8. Conclusion (~0.25 page)

ρ and κ determine when low-bandwidth shared communication adds value in multi-agent collection tasks. The coordination regime is identifiable from two numbers; practitioners can estimate them for their system and determine before deployment whether communication is likely to help.

---

## Appendix

- A: Full per-cell result tables (all 9 cells, all seeds)
- B: Separability fit details (model, residuals, R²)
- C: Parameter translation table (ρ, κ → α, v_max at fixed dt, σ)
- D: Domain mapping estimation methodology
- E: Replication instructions (code, configs, seed schedules)
- F: Connection to Proposition 2 (why count-weighted mean is the sufficient statistic)

---

## Submission checklist

- [ ] Theory section: ρ and κ derivations cross-checked against actual parameter values
- [ ] Pre-registration: analysis exactly matches pre-registered plan in `experiments/grid-design.md`
- [ ] All 9 cells reported, including failures
- [ ] Kill criteria stated and addressed
- [ ] Related work: SMAC, MPE, CommNet, DIAL, Wang et al., Löffler et al. all cited
- [ ] Domain mapping: estimates sourced and defensible
- [ ] Code and configs in appendix sufficient to reproduce all 9 cells
- [ ] SPS-C03 paper cited as prior work (this paper is a follow-on, not a restatement)
