# Business Questions: Coordination Diagnostic Tool

## Core business question

Will robotics and sensor-network engineering teams pay for a tool that isolates whether their inter-agent communication protocol produces coordination or correlation — and is this a problem they currently cannot diagnose with available tools?

## Customer and problem

**Target customer:** Engineering teams that:
- Deploy 3+ agents with inter-agent communication (radio, mesh network, shared memory)
- Use simulation or digital-twin environments for pre-deployment validation
- Have observed unexplained underperformance relative to single-agent baselines or theoretical team predictions

**Industries:** Warehouse robotics, agricultural drones, search-and-rescue multi-robot systems, environmental sensor networks.

**The pain point:** Isaac Sim, Gazebo, AirSim, and CARLA report aggregate task outcomes. They cannot distinguish: (a) communication helping, (b) communication being neutral overhead, (c) communication synchronizing failures. The WO-07 SPS result showed equal-weight sharing decreased performance on half of trials — a gap standard simulators would not surface.

## Validation plan

**Interview target:** 8–10 robotics and sensor-network engineers

**Questions to answer:**
1. Have they observed cases where adding inter-agent communication made performance worse or failed to improve it?
2. When they tested their protocol, did they compare against a shuffled-message or no-message baseline — or only against no-communication?
3. Who makes the communication protocol design decision, and what does it take to change it?

**Success criterion:** ≥3 interviewees describe a real, unresolved case where communication underperformed and they could not diagnose the cause from available metrics.

**Red flag:** All interviewees already test communication vs. no-communication as standard practice. The diagnostic gap we're filling doesn't exist for them.

## Value proposition

**For:** Engineering teams deploying multi-agent systems with communication protocols
**Who need to know:** Whether communication content is doing causal work or just adding correlated noise
**The tool:** Matched counterfactual ablation — runs actual protocol, permuted content, and no content — reports the isolated causal effect of each component
**Unlike:** Standard simulators that only report aggregate outcomes

## Revenue model

| Offering | Price | Buyer |
|---|---|---|
| Open-source ablation runner | Free | Individual engineers, researchers |
| Integration consulting | $8k–20k per engagement | Engineering managers |
| Per-protocol analysis and report | $3k–8k per engagement | Teams needing a deliverable |
| Recurring quarterly diagnostics | $3k–8k / quarter | Operations teams |

**Year 1 target:** 2–5 paid engagements; open-source core for credibility.

## Constraints

- Do not market as a safety or compliance tool — this measures net-positive in simulation, not safety certification.
- Transfer validity is unknown: the approach was validated in the SPS stochastic particle setting. Transfer to non-stationary, heterogeneous, imperfect-communication real-world systems is the core validation risk.
- Paper first. The SPS ablation result is the credibility artifact. No commercial engagement before submission.
