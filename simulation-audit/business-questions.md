# Business Questions: Simulation V&V Audit

---

## Core question

Do simulation engineering teams in aerospace, pharma, finance, or autonomous systems encounter cases where timestep choices, contact-detection models, or event-resolution conventions change decision-critical conclusions — and do they currently lack a systematic way to check?

---

## Customer profiles

### Primary: Simulation QA engineers at mid-size aerospace suppliers
- 50–500 person companies supplying avionics, propulsion, or structural components to primes
- Subject to DO-178C and MIL-STD-882 but can't afford the large consultancies (KPMG, MITRE, Leidos) that dominate full V&V programmes
- They do simulation V&V in-house with general tools (MATLAB, Simulink, Python) and no standard event-audit workflow
- Pain: they need to certify that a first-failure-time conclusion is robust, and they have no structured way to do it

### Secondary: Clinical pharmacology modellers at biotech firms
- Build PK/PD models to support FDA IND/NDA submissions
- Required by FDA guidance to do sensitivity analysis, but event-endpoint sensitivity is not specified
- Pain: when two dosing arms differ by less than one ODE solver timestep at the critical absorption event, the ranking can flip — and this is rarely checked

### Tertiary: Quantitative risk teams at mid-tier banks
- Run internal VaR/CVaR Monte Carlo models for Basel III internal model validation
- Simultaneous-breach event ordering can change net counterparty exposure by material amounts
- Pain: model validation teams audit their models heavily for continuous outputs; the event-ordering layer is reviewed manually at best

### Emerging: Autonomous systems simulation engineers
- Building sim-to-real pipelines for AV, robotics, and UAV certification
- Increasingly aware that sim-to-real gaps often trace to event-handling mismatches
- No standard audit procedure exists; the regulatory frameworks (SOTIF, UL 4600) are still forming

---

## Validation plan

### Interview structure
**Target:** 8–10 engineers across the four segments above (at least 2 from each of aerospace and pharma; 1–2 from finance and AV)

**Opening question (neutral):**
"Walk me through a case where a simulation-based conclusion was later questioned or revised — what caused the revision?"

**Probing questions:**
- "Have you ever changed a timestep or solver tolerance and found that an event-based conclusion changed — which policy was better, which component failed first, which threshold was crossed?"
- "How do you currently document that an event-based result is robust to those kinds of numerical choices?"
- "If you had to certify that your first-failure-time result doesn't flip under any reasonable timestep choice, what would that workflow look like today?"

**Success criterion:** ≥2 interviewees provide documented cases (not hypothetical) where an event-handling or timestep choice changed a conclusion that reached a regulator, customer, or decision-maker.

**Why documented cases, not hypothetical agreement:** Hypothetical agreement ("yes, that sounds like it could be a problem") is not sufficient. The P02 null result shows some simulation systems are already robust. The market exists only where the sensitivity is real and undetected. Documented cases are proof the gap is not already closed.

### Secondary research targets
- FDA warning letters or 483 observations mentioning simulation sensitivity or model-informed dosing
- NTSB reports or DO-178C audit findings mentioning timestep or event-handling
- Published replication failures in AV simulation benchmarks traceable to event-handling
- Basel III internal model validation rejection letters (rare but occasionally public)

---

## Go-to-market path

### Phase 1: Paper and open-source tool (pre-commercialisation)
- Publish the SPS V&V audit methodology in the academic paper (the SPS-P02 methodology section)
- Release an open-source audit runner: given a simulation trace in a standard format, replay under an alternative-convention grid and output a provenance certificate in JSON + PDF
- The tool is free; it generates awareness and surfaces real cases from users
- GitHub stars and issue reports are early market signals — "we ran this on our simulator and found X" is the evidence we need

### Phase 2: First paid engagements (post-paper, post-interviews)
- Target: small aerospace suppliers under DO-178C pressure who found the open-source tool useful but need a formal deliverable for a certification package
- Offer: a structured audit engagement producing a provenance certificate in a format suitable for a V&V report
- Pricing: fixed-scope, deliverable-based (see revenue model below)
- Channel: direct outreach to simulation QA leads at suppliers identified via LinkedIn and conference attendance (AIAA SciTech, IEEE ICSE, SCS Simulation Conference)

### Phase 3: Vertical expansion
- Pharma: partner with a CRO (contract research organisation) that already does FDA submission support; the audit becomes an add-on to their modelling service
- Finance: position as a model validation supplement, not a primary audit; target the model risk management (MRM) function at mid-tier banks
- AV: engage with the emerging SOTIF/UL 4600 standards community; offer audit methodology input as the regulatory framework crystallises

---

## Revenue model

| Offering | Price | Scope | Margin driver |
|---|---|---|---|
| Single-system event-sensitivity audit | $3k–12k | One simulation, one event type, provenance certificate in PDF + JSON | Reusable tooling; most time in setup and documentation |
| Full V&V report with event sensitivity section | $12k–30k | Multiple event types, regulatory format, executive summary | Premium for regulatory framing and signoff |
| Recurring audit subscription | $6k–15k / year | Quarterly re-runs as model evolves; certificate versioning | Predictable; low marginal cost after first engagement |
| Open-source tool support contract | $2k–6k / year | Priority issue response, custom format adapters | Near-zero marginal cost; scales with open-source adoption |

**Year 1 target:** 3–5 engagements, predominantly aerospace, $40k–80k total revenue. This is not a growth-stage target; it is a validation-stage target. Year 1 answers whether the engagements are repeatable and whether clients want ongoing relationships.

**Year 2–3:** If aerospace validates, expand pharma via CRO partnership and autonomous systems via direct outreach. Target 10–15 engagements/year, $150k–300k revenue, 2–3 person operation.

**Ceiling without platform:** Consulting revenue caps at person-hours. The platform play is a SaaS provenance certificate generator with CI/CD integration — runs on every simulation commit, flags event-conclusion drift. That requires a standard trace format and at least 20 pilot clients to define it. Do not design for this before Year 2.

---

## Regulatory hooks (specific, not aspirational)

**DO-178C (aerospace software):** Table A-7 requires verification of outputs; does not specify event-handling sensitivity. An audit certificate that maps event-conclusions to tested numerical assumptions is a defensible addendum to a DO-178C V&V package. Requires legal review before claiming compliance.

**FDA Model-Informed Drug Development guidance (2019):** Section IV.B requires "assessment of model uncertainty and sensitivity." Event-endpoint sensitivity is not excluded, but the guidance focuses on continuous PK outputs. An audit structured around the guidance's sensitivity requirements is defensible as supplemental documentation. Requires FDA-experienced regulatory counsel before submission.

**Basel III FRTB internal model validation:** Article 325bh requires stress testing of model assumptions. Timestep and event-ordering conventions qualify as model assumptions. An audit certificate can be framed as a model assumption stress test. Requires review by the firm's model risk management function.

**SOTIF (ISO 21448) and UL 4600 (autonomous systems):** Both are process standards, not prescriptive test specifications. They require evidence of systematic safety analysis. An event-sensitivity audit is a direct fit for the "systematic analysis of behavioral competencies" requirement in SOTIF. These standards are still forming — early engagement with the standards bodies is a market-positioning opportunity, not just a compliance play.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| External documented cases don't materialise after 10 interviews | Medium | High — kills idea | Expand to secondary research (FDA warning letters, NTSB reports) before declaring no market; don't kill after 0 cases |
| P02 null result generalises — most simulation systems are already robust | Medium | High — market is smaller than expected | Reframe product as "certification that you are robust" not "fixing sensitivity"; value is in the certificate, not the finding |
| Regulatory framing requires legal clearance that is slow/expensive | Medium | Medium — delays first engagement | Keep first engagements in internal V&V (not regulatory submission) to avoid legal barrier while building credibility |
| Open-source tool attracts users who never convert to paid | Low | Low — expected | Open-source is awareness, not revenue; conversion requires a formal deliverable need (certification, submission, audit) that the free tool can't satisfy alone |
| Large consultancies (MITRE, Leidos) copy the methodology | Low | Medium | Speed-to-credibility matters; being first with a published methodology and an open-source implementation creates a moat that is hard to copy quickly |
| Standard trace format doesn't exist; integration cost is prohibitive | High | Medium — limits addressable market | Scope Phase 1 to Python/MATLAB/Simulink traces only; don't promise cross-platform until format standardisation is underway |

---

## Hard constraints

- The P02 null result means we cannot use our own system as a positive case study. Every client conversation requires external evidence.
- Scope must be bounded tightly. Provenance certificates cover only the event types and convention dimensions explicitly tested — they cannot be presented as comprehensive V&V.
- Any regulatory framing (DO-178C, FDA, Basel) requires jurisdiction-specific legal and regulatory review before use in a submission.
- Paper first. The SPS-P02 methodology section is the scientific and credibility foundation. No paid engagements before submission.
- Do not build the SaaS platform before 20 pilot clients have validated the trace format and certificate structure. Build consulting first.
