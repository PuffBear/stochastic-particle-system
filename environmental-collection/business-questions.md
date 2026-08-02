# Business Questions: Environmental Collection Planning

## Core business question

Will environmental consultancies and program operators pay for a decision layer that quantifies whether autonomous mobile collectors are worth the capital and operational cost over passive fixed sensors — and is this decision currently made without quantitative support?

## Customer and problem

**Target customer:** Environmental engineers and consultancies running:
- Autonomous water-quality or contaminant sampling programs
- Oil-spill or marine debris recovery operations
- Sediment, microplastics, or harmful algal bloom monitoring deployments

**Decision context:** Evaluating whether to procure autonomous mobile collectors (drones, USVs, AUVs) vs. deploying a fixed sensor array or passive buoys. This decision is made 3–6 months before program deployment with capital consequences of $50k–$5M depending on the program.

**The gap:** EFDC, WASP, and Delft3D model how material moves through the environment. They do not model the marginal value of a collector that can move toward the material. The mobility procurement decision is answered today by analogy or engineering intuition — not simulation.

## Validation plan

**Interview target:** 6–8 environmental engineers, program managers, and consultants

**Questions to answer:**
1. Do they currently have a quantitative method for the mobility procurement decision, or is it heuristic?
2. Is flow field data available in a format that could feed a planning tool (EFDC output, sensor arrays, satellite current data)?
3. Who owns the procurement decision — engineering manager, program director, or committee?
4. What would a quantitative planning report need to contain to influence that decision?

**Success criterion:** ≥3 interviewees confirm they make the mobility decision without quantitative analysis and would value a planning tool computing expected collection improvement from mobility.

**Gating validation:** Before any commercial engagement, obtain one authorized historical dataset where both mobile and stationary systems were deployed in the same area. Compute κ from the flow field and check whether the mobility advantage matches the SPS-derived prediction. Binary gate: pass = proceed, fail = κ parameterization needs domain calibration.

## Value proposition

**For:** Environmental engineers planning monitoring or recovery programs
**Who face:** The procurement question — mobile vs. passive collector systems
**The tool:** κ-based catchability analysis + simulated collection comparison using the program's existing flow field model as input
**Unlike:** EFDC/WASP, which model transport but not collector decision value
**Deliverable:** A catchability map and planning report: "At your flow conditions, mobile collection is expected to improve yield by X% over passive, with highest-value zones here."

## Revenue model

| Offering | Price | Scope |
|---|---|---|
| Pre-deployment feasibility study | $15k–40k | Uses client's existing flow model; 120–300 hours |
| Catchability map + planning report | $8k–20k | Focused report for one deployment area |
| On-call planning advisory | $4k–8k / quarter | For programs with recurring seasonal deployments |

**Year 1 target:** 1–2 feasibility studies; highest value at the pre-deployment planning stage.

## Constraints

- Do not claim pollutant removal capability — regulatory remediation filings require certified methods. This is a planning aid, not a certified environmental assessment.
- Transfer validity is the primary blocker — SPS uses a simplified uniform field. Real flow fields are spatially heterogeneous and time-varying. The κ parameterization may need domain-specific calibration.
- Do not quote accuracy without a validated case study. All estimates must be framed as scenario comparisons (mobile vs. stationary under modeled conditions), not as absolute recovery predictions.
- Paper first. The SPS catchability result is the scientific foundation. No commercial application before paper submission.
