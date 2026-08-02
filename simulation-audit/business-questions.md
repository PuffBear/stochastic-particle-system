# Business Questions: Simulation V&V Audit

## Core business question

Do simulation engineering teams in aerospace, pharma, or finance encounter cases where timestep choices or event-handling conventions change decision-critical conclusions — and are those teams currently unable to detect or diagnose this sensitivity?

## Target customers and problems

**Aerospace and defense:**
- First-failure time is a safety-critical simulation output (MIL-STD-882, DO-178C)
- V&V standards (ASME V&V 20, NIST Digital Twin credibility framework) require verification of conclusions but don't specify a workflow for event-specific sensitivity
- Gap: "the state trajectories converge" (standard V&V) is not the same as "the policy ranking is robust"

**Pharmaceutical:**
- PK/PD simulations with absorption-event endpoints
- Timestep sensitivity in ODE solvers is a known issue in FDA submissions
- Policy rankings (dosing schedule comparisons) can shift under different solver tolerances

**Financial risk:**
- Monte Carlo models where threshold-crossing (margin call, VaR breach, default) drives decisions
- Event-handling conventions (e.g., how simultaneous breaches are ordered) can affect reported risk numbers
- Increasing regulatory documentation requirements (SEC, FINRA, Basel committee)

## Validation plan

**Interview target:** 8–10 simulation engineers across aerospace, pharma, and finance

**Questions to answer:**
1. Have they encountered cases where changing a numerical parameter (timestep, solver tolerance, event rule) changed a decision-critical conclusion?
2. If yes: was the sensitivity discovered accidentally or through systematic testing?
3. What format does a V&V deliverable need to satisfy internal or regulatory reviewers?
4. What is the unit of work — a single model, a single scenario, a full V&V package?

**Success criterion:** ≥2 interviewees provide documented cases (not hypothetical) where event-handling or timestep choices changed a conclusion that mattered to a regulator, customer, or decision-maker.

**Why two documented cases, not hypothetical agreement:** The SPS-P02 null result shows not all simulation systems are sensitive to event-handling. Hypothetical agreement ("yes, that sounds like it could be a problem") is not sufficient — the market exists only where the sensitivity is real.

## Value proposition

**For:** Simulation QA teams and V&V engineers
**Who need to demonstrate:** That event-based conclusions are robust to numerical assumptions
**The audit:** Structured replay under alternative timestep, contact-model, and tie-resolution conventions; provenance certificate documenting which conclusions survived
**Unlike:** General sensitivity analysis (varies continuous parameters) or state-trajectory convergence tests (don't target event-specific conclusions)

## Revenue model

| Offering | Price | Scope |
|---|---|---|
| Single-system event-sensitivity audit | $3k–12k | One simulation, one event type, provenance certificate |
| Full V&V report with event sensitivity | $12k–30k | Multiple event types, regulatory format |
| Recurring audit subscription | $6k–15k / year | Quarterly re-runs as model evolves |

**Year 1 target:** 3–5 engagements across aerospace, pharma, and finance.

## Constraints

- The P02 null result is a caution signal: we cannot use our own research as a positive case study. Credibility rests entirely on external documented cases.
- Scope must be bounded tightly. A provenance certificate covers only the specific events and conventions tested — do not present it as a comprehensive V&V audit.
- Regulatory framing requires legal review before claiming any standard (DO-178C, FDA guidance, Basel III) is satisfied.
- Paper first. The SPS V&V audit methodology (SPS-P02) is the scientific foundation. No commercial application before submission.
