# FI-1 Coordination Diagnostic Tool — Sequential Plan

**Goal:** Build and validate an open-source tool that answers the question "Is our communication protocol helping or hurting?" for multi-robot and multi-sensor teams. Reach first paid engagement within 12 months of paper submission.

---

## Why this plan is sequenced the way it is

The three-condition diagnostic (actual / permuted / no messages) already exists as a methodology — WO-07C ran exactly this structure. The tool is not a research bet; it is an engineering and validation bet. The sequence is:

1. Formalise the methodology as a reusable specification (no new experiments needed)
2. Build a working prototype against the SPS traces we already have (no new data needed)
3. Validate market demand through interviews (no customers needed yet)
4. Run one pro-bono engagement to stress-test the workflow (no revenue yet)
5. First paid engagement

Steps 1–2 can run in parallel. Step 3 starts as soon as the paper is submitted (credibility anchor). Step 4 requires ≥3 documented failure cases from Step 3. Step 5 follows from Step 4.

---

## Phase 0: What we already have (no work required)

| Asset | Status |
|---|---|
| Three-condition methodology | ✅ Implemented in WO-07C |
| Positive case (v2 > shuffled, 4/8 seeds) | ✅ Documented |
| Negative case (v1 fails the diagnostic) | ✅ Documented |
| SPS trace format (JSONL episode summaries) | ✅ Exists |
| Statistical framework (paired sign test) | ✅ Used in C03 |

We are not starting from zero. We are packaging what already works.

---

## Phase 1: Methodology specification (Weeks 1–2)

**Deliverable:** A formal, environment-agnostic specification of the three-condition diagnostic that a simulation engineer can read and implement against their own system without knowing anything about SPS.

**Files to produce:**
- `methodology/diagnostic-spec.md` — inputs, conditions, outputs, assumptions
- `methodology/statistical-framework.md` — how to interpret gain-from-content vs gain-from-structure
- `methodology/certificate-format.md` — JSON schema for the provenance certificate output

**Why this comes first:** The spec is what makes the prototype credible. Without a formal spec, the prototype is just a script. With it, the prototype is an implementation of a documented methodology — which is what reviewers, interviewees, and eventual clients need to see.

---

## Phase 2: Prototype diagnostic tool (Weeks 2–4, parallel with Phase 1)

**Deliverable:** A working Python tool (`diagnostic.py`) that accepts simulation traces in a standard format, runs the three-condition comparison, and outputs a structured provenance certificate.

**Design constraints:**
- Input format: simple JSON (not SPS-specific; adapters for common formats)
- Output: machine-readable JSON certificate + human-readable summary
- No ML dependencies; pure NumPy/SciPy
- Single-file core; easily auditable

**Validation:** Run against SPS WO-07C traces. The tool must reproduce the documented WO-07C numbers (+0.875 v2 vs shuffled, +1.625 shuffled vs independent) to within floating point.

**Why a working prototype matters:** In every interview and every early engagement, the question will be "can I try this on my system?" A working tool with a standard input format makes the answer "yes, here's how." A spec document alone doesn't.

---

## Phase 3: Interview validation (Weeks 4–8, starts after paper submission)

**Deliverable:** 8–10 structured interviews with robotics and sensor-network engineers. Documented cases, not hypothetical agreement.

**Target interviewees:**
- ROS community contributors (search GitHub/ROS Discourse for multi-robot coordination repos with active issues about communication)
- Agricultural drone teams (DJI enterprise, Sentera, PrecisionHawk — known multi-agent coordination problems)
- Search-and-rescue robotics researchers (DARPA Subterranean Challenge teams)
- Environmental sensor network engineers (USGS, NOAA instrument networks)
- Warehouse robotics engineers (mid-size integrators, not Amazon — Amazon builds in-house)

**Success gate:** ≥3 interviewees provide documented failure cases (not hypothetical) where they could not determine whether their communication protocol was helping or hurting.

**If the gate fails:** Do not build integrations. The methodology is still a paper contribution (matched-counterfactual communication audit); the commercial path closes.

---

## Phase 4: Pro-bono pilot engagement (Weeks 8–16, after gate passes)

**Deliverable:** One complete engagement — from trace collection to provenance certificate — with a partner identified from Phase 3 interviews. No charge; the deliverable is a case study we can reference.

**What to learn from the pilot:**
- Is the trace format simple enough for a non-SPS team to produce?
- Is the certificate output legible to an engineering manager (not just the simulation engineer)?
- What questions does the certificate *not* answer that the client needs answered?
- How long does the engagement actually take?

**Success criterion:** The partner confirms the certificate would have been useful at a specific past decision point — and says so in writing (for use as an anonymous case study).

---

## Phase 5: First paid engagement (Month 4–6 post-paper)

**Target:** One aerospace or robotics team, $5k–12k, delivering a provenance certificate suitable for an internal V&V report.

**Positioning:** "Coordination audit" not "communication testing." The word "audit" signals a deliverable, a scope, and a professional process. "Testing" implies the client runs it themselves.

**What the engagement includes:**
1. Trace format integration (adapting the client's simulation logs to the diagnostic input)
2. Three-condition run + statistical analysis
3. Provenance certificate (JSON + PDF summary)
4. One-hour walkthrough with the team

**What it does not include:** Recommendations for fixing communication (that's a separate engagement). The audit tells you whether your protocol is helping; it does not redesign it.

---

## Reviewer-facing framing

If this were being evaluated by an accelerator or a technical due diligence reviewer, the claim structure is:

1. **The methodology is already validated** — WO-07C ran the three-condition test in a controlled environment and reproduced a known effect. This is not a promise; it is a demonstration.

2. **The competitive gap is structural** — no existing tool (Isaac Sim, Gazebo, AirSim, ROS diagnostics) runs matched counterfactuals. They report aggregate reward; they cannot attribute it to communication.

3. **The market is identifiable** — not "companies that do robotics" but specifically teams that have deployed multi-agent communication and have no systematic way to validate it. That's a definable segment.

4. **The failure mode is documented** — v1 failed the diagnostic in a specific, diagnosable way. This is industrially relevant: teams that are coordinating on noise will see the same failure pattern. The tool catches it.

5. **The ask is small** — open-source core, consulting revenue model, no upfront capital required. The risk is interview validation failing. That risk resolves in 8 weeks.

---

## What this plan does not promise

- It does not promise a SaaS product. That requires 20+ pilot clients and a standard trace format that doesn't exist yet.
- It does not promise regulatory use. Provenance certificates are internal V&V documents, not compliance filings.
- It does not promise the interviews will validate the market. The gate is real.
- It does not promise the paper needs to be submitted first — but the paper is the credibility anchor for every interview. Starting interviews before submission is possible but will be harder.
