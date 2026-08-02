# Simulation V&V Audit for Event-Driven Systems

**Industry idea:** FI-3
**Status:** Unvalidated — requires external documented failure cases before building
**Readiness:** Low (but higher than currently written; see reframing below)

---

## The problem

Standard simulation V&V asks: *do the state trajectories converge under different numerical assumptions?* This is the right question for continuous outputs. It is the wrong question when the engineering conclusion depends on an **event** — a first contact, a threshold crossing, a failure occurrence, a default trigger.

Events have a different sensitivity profile than state trajectories:

- An event can shift in *time* under a different timestep — changing whether a deadline is met or a safety margin is violated
- An event can change *ownership* under a different contact-detection model — determining which agent, component, or counterparty is credited or liable
- An event can *reverse in rank* under a different tie-resolution rule — flipping a policy comparison that drove a procurement or dosing decision

None of the standard V&V workflows (ASME V&V 20, NASA-STD-7009, FDA guidance on model-informed drug development) specify a procedure for testing event-conclusion robustness. They require trajectory convergence and uncertainty quantification over continuous outputs. The event layer is structurally unaddressed.

---

## Where the gap shows up

**Aerospace and defense**
- First-failure-time in hardware fault-tree simulations (MIL-STD-882 hazard analysis)
- Thermal threshold-crossing in spacecraft or electronics simulations (when temperature exceeds a limit is the decision, not what the temperature is)
- Collision detection timing in AV or UAV certification simulations (DO-178C requires V&V but doesn't specify event-handling conventions)
- Wargaming simulations where first-kill or first-contact determines doctrine conclusions

**Pharmaceutical and biotech**
- Absorption-event endpoints in PK/PD models: when drug reaches Cmax or crosses a target concentration determines dosing interval comparisons
- Event-driven dosing in clinical trial simulations: which arm delivers first can determine protocol rankings under different ODE solver tolerances
- FDA guidance on model-informed drug development explicitly requires sensitivity analysis, but treats event endpoints the same as continuous outputs — no specific workflow exists

**Financial risk**
- Monte Carlo VaR and CVaR models where margin call or default events drive counterparty exposure calculations
- The ordering of simultaneous breach events under different timestep conventions can change reported net exposure
- Basel III internal model validation requirements are stringent but event-ordering is an unspecified gap

**Autonomous systems (emerging)**
- Simulation-to-real transfer failures increasingly trace back to event-handling mismatches between training simulator and deployment: contact timing, collision ownership, signal threshold crossing
- No standard audit procedure exists for this class of failure

---

## What the audit produces

Given a simulation trace with logged events, the audit replays the same scenario under a structured grid of alternative numerical assumptions and records:

1. **Time sensitivity:** do event timestamps shift under alternative timestep choices, and by how much?
2. **Ownership sensitivity:** does event ownership (agent, component, counterparty) change under alternative contact or detection models?
3. **Rank sensitivity:** do policy or design rankings reverse under any tested convention?

**Output:** a *provenance certificate* — a structured document mapping each reported conclusion to the set of numerical assumptions it survived. Suitable for internal V&V reports, regulatory submissions, and procurement documentation.

The certificate is bounded: it covers only the event types and convention dimensions tested. It does not claim the simulation is globally correct; it claims the specific conclusion is robust to the specific set of numerical alternatives tested.

---

## The P02 result reframed

The SPS project's own event-sensitivity audit (SPS-P02) found that 0/144 first-interception outcomes changed when exact contact mathematics were substituted. This is routinely described in our notes as a "caution signal" — but that framing is wrong for commercial purposes.

The correct framing: **the SPS-P02 result validates the audit methodology, not the absence of market**. It shows the audit can be run, produces a credible certificate, and in this case the system passed. The market exists in the systems that *would not* pass. The value of the tool is the ability to check — not the expectation of finding failures.

An analogy: the fact that a structural engineer's building passed a load test doesn't mean load testing has no market. It means the test worked and the building is certified.

What the P02 result does legitimately constrain: we cannot use our own research as a positive case study. We need external documented cases where event-handling sensitivity changed a conclusion that mattered.

---

## Competitive gap

| Tool / approach | What it checks | What it misses |
|---|---|---|
| Dakota, OpenTURNS | Continuous parameter sensitivity | Event-handling conventions |
| Monte Carlo convergence tests | State distribution convergence | Event ordering and ownership |
| ASME V&V 20 workflow | State trajectory uncertainty | Event-conclusion robustness |
| Internal QA / code review | Correctness of implementation | Sensitivity of conclusions to valid alternatives |
| General sensitivity analysis | Continuous output variance | Discrete event occurrence and rank |

No existing tool specifically targets the event-conclusion layer. The gap is structural, not a market oversight.

---

## Files

- `business-questions.md` — validation plan, go-to-market, revenue model, risk register, regulatory hooks
