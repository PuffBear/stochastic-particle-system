# FR-B3 Catchability Benchmark — Full Publication Plan

**Target venue:** ICML 2027 (primary) / NeurIPS 2027 D&B track (fallback)
**Submission deadline:** ICML ~February 2027 | NeurIPS ~February 2027
**Realistic timeline from now:** 5 months to submission-ready draft

---

## The paper in one paragraph

We introduce a nondimensional parameterization of multi-agent collection tasks using two quantities: ρ (sensing difficulty, how hard it is to detect the signal) and κ (control authority, how fast collectors move relative to the signal). We show empirically that the coordination gain from a minimal shared communication channel is approximately separable in ρ and κ — meaning its value can be predicted from these two numbers alone, independent of other simulation details. We run a 3×3 grid across (ρ, κ) space, with the SPS-C03 confirmed result as the anchor cell, and test whether the gain pattern matches the theoretical prediction. The output is both a theoretical framework and a practical benchmark: given any multi-agent collection system, compute its (ρ, κ), find the grid cell, and read off whether shared communication is likely to help.

---

## Why ICML, not AAMAS or NeurIPS D&B

**ICML (primary):**
ICML 2027 is the right venue if the separability claim is theoretically grounded — meaning we either prove it from first principles or provide a mechanistic argument for why ρ and κ should appear as independent factors. The ρ-κ framework has the right shape for ICML: a dimensionless parameterization derived from physics, tested empirically, with a falsifiable prediction. ICML reward this structure.

**NeurIPS D&B (fallback):**
If the empirical grid is the lead contribution and the theory is supporting, NeurIPS 2027 Datasets & Benchmarks is the natural home. D&B papers are evaluated on whether the benchmark enables future work — and a parameterized environment where every (ρ, κ) cell is reproducible does that. This is the safer submission but the less prestigious one for this specific contribution.

**Why not AAMAS:**
AAMAS is the right home for the coordination mechanism papers (FR-A1, FR-A2). This paper's contribution is a parameter space and a separability claim — that's more ICML/NeurIPS than AAMAS, which focuses on agent behaviour rather than task parameterization.

**Why not ICLR:**
The paper doesn't have a representation learning or deep learning core. ICLR reviewers would ask "where's the neural network?" and the answer here is "there isn't one."

---

## Phases

### Phase 0: What we already have (no work required)

| Asset | Status |
|---|---|
| SPS-C03 confirmed result at (ρ≈0.21, κ≈0.20) | ✅ One anchor cell, +1.19 particles |
| shared_summary_v2 controller | ✅ Implemented and confirmed |
| capacity_matched_independent controller | ✅ Implemented and confirmed |
| Matched counterfactual infrastructure | ✅ Pre-generated noise tensors |
| Scripted baselines (stationary, random, greedy, oracle) | ✅ All implemented |

We are filling in 8 cells around a confirmed centre point — not starting a new experiment from scratch.

---

### Phase 1: Theory — derive and formalise ρ-κ (Weeks 1–2)

**Deliverable:** `theory/parameterization.md` — a complete derivation of ρ and κ from first principles, their predicted effect on coordination gain, and the separability hypothesis.

**Key tasks:**
- Derive ρ = α·√dt / σ from the signal detection framing (SNR of the field estimate)
- Derive κ = α / v_max from the catchability framing (can a collector outrun a particle?)
- State the separability hypothesis: Δ̄(ρ, κ) ≈ g(ρ) · h(κ)
- Predict the qualitative shape of g and h (both should be monotone increasing; g concave, h linear or sublinear)
- Connect to Péclet number (Pe = v·L/D in fluid dynamics — κ is analogous)
- Connect to catchability in ecology literature (Beverton-Holt, fisheries models)
- State kill criterion: if the fitted surface has residuals comparable to signal, separability is false and the paper's central claim collapses

**Why this comes first:** The experiment grid must be designed to test the theory, not the other way around. If the grid is designed before the theoretical prediction is written down, the paper looks exploratory rather than confirmatory.

---

### Phase 2: Pre-registration and grid design (Week 2)

**Deliverable:** `experiments/grid-design.md` — the exact 3×3 parameter grid, seed plan, and pre-registered analysis.

**The grid:**

| | κ = 0.10 (low authority) | κ = 0.20 (mid, confirmed) | κ = 0.40 (high authority) |
|---|---|---|---|
| **ρ = 0.10** (easy sensing) | Cell (1,1) | Cell (1,2) | Cell (1,3) |
| **ρ = 0.21** (mid, confirmed) | Cell (2,1) | Cell (2,2) ✅ C03 | Cell (2,3) |
| **ρ = 0.35** (hard sensing) | Cell (3,1) | Cell (3,2) | Cell (3,3) |

**Parameter translations** (at fixed dt=0.02, σ=0.06, M=4, N=256, steps=67):

| ρ | α |
|---|---|
| 0.10 | 0.028 |
| 0.21 | 0.060 (confirmed) |
| 0.35 | 0.099 |

| κ | v_max |
|---|---|
| 0.10 | α / 0.10 |
| 0.20 | α / 0.20 (confirmed) |
| 0.40 | α / 0.40 |

Note: κ is defined per cell (κ = α / v_max), so v_max varies with α across the grid. Each cell specifies (α, v_max) jointly.

**Seed plan:**
- 8 diagnostic seeds per cell (72 runs total) — go/no-go
- 16 confirmatory seeds per cell if diagnostic passes (144 additional runs)
- SPS-C03 seeds (1001–1032) are reused for cell (2,2); no new runs needed there

**Pre-registered analysis:**
- Primary: sign count ≥5/8 for Δ = Y(shared_v2) − Y(independent) in each cell
- Secondary: fit multiplicative model Δ̄(i,j) = a · g_i · h_j, compute R²
- Kill criterion: R² < 0.5 on the multiplicative fit after 9 cells (separability rejected)
- Exploratory: oracle gap per cell (how much headroom remains)

---

### Phase 3: Run the experiments (Weeks 3–5)

**Compute requirements:**
- 8 diagnostic seeds × 8 remaining cells × 2 arms × ~10s/episode = ~21 hours single-threaded
- Trivially parallelisable; 4 cores reduces to ~5 hours
- No HPC required — this runs on a laptop or a small cloud instance

**Implementation:**
- Modify the SPS config to accept (α, v_max) as experiment parameters
- Run the existing `run_sps_c03_confirmation.py` script with the new config for each cell
- Reuse all existing infrastructure: matched noise tensors, episode logging, manifest format

**Quality checks per cell:**
- Stationary baseline must be lower than independent (sanity check)
- Oracle must be highest (sanity check)
- Seed variance within expected range (SD < 5.0)

---

### Phase 4: Analysis and separability test (Week 5–6)

**Deliverable:** `experiments/results.md` + fitted surface plot

**Analysis steps:**
1. Compute Δ̄(i,j) for each of 9 cells (mean paired difference, shared_v2 vs independent)
2. Fit multiplicative model: Δ̄(i,j) = a · exp(b·ρ_i) · exp(c·κ_j) using least squares
3. Compute R² of the fit — primary separability test
4. Plot the 3×3 heatmap of observed Δ̄ and the fitted surface side-by-side
5. Compute oracle gap per cell: oracle − shared_v2
6. Identify the "coordination regime": cells where Δ̄ > 1.0 and sign count ≥6/8

**Expected qualitative pattern:**
- High ρ + high κ: large positive Δ (hard to sense, can catch — communication most valuable)
- Low ρ + low κ: near-zero Δ (easy to sense + can't catch — communication irrelevant)
- Cell (2,2) anchored at +1.19 (confirmed)

**If pattern is unexpected:**
- Low ρ giving high Δ: collectors are using communication to compensate for their own blindness — revisit the mechanism interpretation
- High κ giving low Δ: collectors are fast enough that they don't need coordination — possible if v_max >> α

---

### Phase 5: Domain mapping (Week 6)

**Deliverable:** `theory/domain-mapping.md` — table showing where real systems fall in ρ-κ space

**Target domains and their (ρ, κ) estimates:**

| Domain | System | ρ estimate | κ estimate | Notes |
|---|---|---|---|---|
| Microplastics monitoring | AUV + ocean current | 0.15–0.25 | 0.1–0.3 | AUV speed vs drift speed |
| Agricultural drone swarms | UAV + pest diffusion | 0.05–0.15 | 0.3–0.8 | Fast drones, slow pests |
| Search-and-rescue | Ground robot + person movement | 0.2–0.4 | 0.5–2.0 | People move faster than robots in rubble |
| Oil spill cleanup | Autonomous vessel + slick drift | 0.1–0.2 | 0.2–0.5 | Slick moves with wind |
| Financial surveillance | Monitor + anomaly diffusion | N/A (abstract) | N/A | ρ-κ applies only to physical diffusion |

**Purpose in the paper:** The domain mapping section is what makes the paper useful to practitioners and what ICML reviewers remember. It answers "so what?" — the benchmark tells you whether communication will help *before* you deploy.

---

### Phase 6: Writing (Weeks 6–10)

**Target length:** 8 pages + appendix (ICML format)

**Section outline:**

1. **Introduction** (1 page)
   - The unsolved problem: when does inter-agent communication add value in collection tasks?
   - The key insight: two nondimensional numbers determine the answer
   - Main result: coordination gain is approximately separable in ρ and κ
   - Practical payoff: the benchmark predicts regime before deployment

2. **Related work** (0.75 page)
   - Multi-agent coordination benchmarks (SMAC, MPE, Cooperative Navigation) — none parameterize sensing difficulty and control authority jointly
   - Péclet number in fluid mechanics — analogous dimensionless ratio
   - Catchability in ecology (Beverton-Holt) — κ maps directly
   - Prior work on communication value (Wang et al. 2025, Löffler et al. 2023) — single-collector or no controlled comparison

3. **The ρ-κ parameterization** (1 page)
   - Derive ρ from signal detection framing
   - Derive κ from catchability framing
   - State separability hypothesis
   - Predict qualitative shape of g(ρ) and h(κ)

4. **The SPS benchmark** (1 page)
   - Environment description (minimal — readers of FR-A papers already know it)
   - Experimental protocol: matched counterfactual, pre-registered grid
   - Shared_summary_v2 vs capacity_matched_independent: the comparison

5. **Results** (2 pages)
   - 3×3 heatmap of Δ̄ (observed vs fitted)
   - Separability: R², residual analysis
   - Oracle gap per cell
   - The coordination regime: where ρ-κ puts you in "communication helps"

6. **Domain mapping** (0.75 page)
   - Table of real systems with estimated (ρ, κ)
   - Practical guidance: if your system is in this region, communication is likely worth implementing

7. **Limitations and future work** (0.5 page)
   - Uniform field assumption
   - Scripted (not learned) policies
   - 2D symmetric arena
   - FR-B1/B2 extensions (learned policies, √M scaling)

8. **Conclusion** (0.25 page)

**Appendix:**
- Full per-cell results tables
- Parameter translation details
- Proof sketches for the separability prediction
- Replication instructions

---

### Phase 7: Review and submission (Weeks 10–12)

**Internal review checklist:**
- [ ] Theory section: ρ and κ derivations correct and referenced
- [ ] Pre-registration: analysis matches pre-registered plan
- [ ] All 9 cells reported, including any that failed the diagnostic gate
- [ ] Kill criteria addressed: if separability was rejected, say so and redirect claim
- [ ] Related work: Wang et al. and Löffler et al. correctly positioned
- [ ] Domain mapping: estimates are defensible, not fabricated
- [ ] Replication: code and configs sufficient to reproduce all 9 cells

**Submission target:** ICML 2027 (deadline ~January 31, 2027)

**Fallback:** NeurIPS 2027 D&B (deadline ~February 2027, different track)

---

## What this paper does NOT claim

- It does not claim that ρ-κ is sufficient to determine whether communication helps in any system — only in systems that structurally resemble the SPS task (diffusion + mobile collection)
- It does not claim the 3×3 grid is exhaustive — it is diagnostic; a full surface requires more cells
- It does not claim the shared_summary_v2 controller is optimal — the oracle gap shows substantial headroom remains
- It does not claim learned policies would show the same pattern — FR-B1/B2 tests that

---

## Kill criteria

Stop or redirect if:
- The multiplicative fit R² < 0.5 across 9 cells (separability claim collapses)
- Fewer than 5/9 cells show positive Δ̄ (coordination is not reliably beneficial across the regime)
- Cell (2,2) does not replicate C03 result within ±0.5 particles (infrastructure problem)
- Domain mapping finds no plausible physical systems in the positive coordination regime
