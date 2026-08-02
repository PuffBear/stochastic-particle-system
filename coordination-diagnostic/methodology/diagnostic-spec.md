# Coordination Diagnostic: Formal Specification

**Version:** 0.1
**Status:** Draft — validated against SPS WO-07C; not yet validated on external systems

---

## Purpose

This specification defines the three-condition coordination diagnostic: a structured procedure for determining whether a multi-agent communication protocol is adding value beyond its structural format, or whether the team is coordinating on noise.

The diagnostic is environment-agnostic. It requires only that the target system can:
1. Log inter-agent messages as part of a simulation trace
2. Replay a trace under alternative message conditions
3. Report a scalar team performance metric per episode

---

## Definitions

**Episode:** A single run of the multi-agent task from a fixed initial state to a fixed terminal condition. Episodes are identified by a seed that determines initial state and stochastic disturbances.

**Message:** Any value transmitted from one agent to another during an episode. Messages may be vectors, scalars, or structured objects. The diagnostic treats message content as opaque — it does not require knowledge of what messages mean.

**Team performance metric Y:** A scalar summary of team performance for an episode. Must be:
- Higher = better (negate if lower = better)
- Comparable across episodes with the same seed
- Unaffected by which condition is running (no information leakage from condition into environment dynamics)

**Matched seed pair:** Two episodes run with the same seed but under different message conditions. The matched structure eliminates trajectory variance as a confound.

---

## The three conditions

### Condition A: Actual messages
Agents send and receive their real messages as produced by the communication protocol under evaluation. This is the system as deployed.

### Condition B: Permuted messages
At each timestep, the messages received by each agent are randomly permuted across agents — each agent receives a message that was sent by *some* agent, but not necessarily the one it was addressed to. The message format (dimensionality, value range) is preserved; only the assignment is scrambled.

**What permutation isolates:** Condition B preserves the structural effect of receiving a signal in the expected format while removing the content correspondence between sender state and receiver. If B performs comparably to A, the content of the messages is not doing useful work — only the format matters.

### Condition C: No messages
Agents receive null messages (zeros, or a fixed baseline value). The action policy and all other environment dynamics are unchanged.

**What no-messages isolates:** Condition C is the independent baseline. The gap between C and B measures the value of having *any* shared signal in the expected format. The gap between B and A measures the additional value of correct content.

---

## The two estimands

Let Y_s(X) be the team performance metric in episode with seed s under condition X ∈ {A, B, C}.

**Gain from content:**
```
Delta_content(s) = Y_s(A) - Y_s(B)
```
Positive: correct message content adds value beyond format alone.
Zero or negative: the protocol is coordinating on noise — format accounts for all or more than all observed gain.

**Gain from structure:**
```
Delta_structure(s) = Y_s(B) - Y_s(C)
```
Positive: having any shared signal in the expected format helps, even with scrambled content.
Zero or negative: the communication channel adds no value even as a structural scaffold.

**Net coordination value:**
```
Delta_net(s) = Y_s(A) - Y_s(C) = Delta_content(s) + Delta_structure(s)
```

---

## Required inputs

### Trace format
Each episode produces a trace file containing:

```json
{
  "seed": 1001,
  "condition": "actual",
  "steps": [
    {
      "t": 0,
      "messages": [
        {"sender": 0, "receiver": 1, "value": [0.12, -0.34, 0.88]},
        {"sender": 1, "receiver": 0, "value": [-0.05, 0.21, 0.73]}
      ],
      "team_state": "..."
    }
  ],
  "outcome": {
    "metric": 11.0,
    "metadata": {}
  }
}
```

Minimum required fields: `seed`, `condition`, `outcome.metric`. Message logging is required only for Condition A; Conditions B and C can be derived from A's message log.

### Seed set
A minimum of 8 matched seeds is required for the diagnostic gate. 16 seeds are recommended for reliable sign counts. The confirmatory standard (if proceeding to a formal claim) requires 32 seeds with a pre-registered one-sided bootstrap lower bound.

---

## Required assumptions

1. **Matched initialization:** Episodes with the same seed must start from identical initial states across all three conditions.
2. **Matched stochasticity:** Post-initialization stochastic disturbances must be drawn from the same pre-generated sequence across conditions. If the system uses an RNG, it must be seeded identically and consumed in the same order through the point of message divergence.
3. **No leakage:** The condition label must not influence environment dynamics, reward signals, or observations beyond the message channel being tested.
4. **Stationary protocol:** The communication protocol must not adapt during the diagnostic run. If the protocol is learned (e.g. trained with communication), it must be frozen before the diagnostic runs.

Assumption 3 is the most commonly violated in practice. Systems that route agents based on message content, or that use message presence as a trigger for environment events, will produce confounded results.

---

## Diagnostic gate (8-seed version)

The diagnostic gate is descriptive. It does not support a formal statistical claim but is sufficient for go/no-go decisions in an engagement.

**Positive content signal:** sign(Delta_content(s)) > 0 for ≥5/8 seeds.
**Positive structure signal:** sign(Delta_structure(s)) > 0 for ≥5/8 seeds.

A protocol that passes the content gate is adding value beyond its format. A protocol that fails the content gate but passes the structure gate is coordinating on format, not information — a diagnosable and fixable failure mode.

A protocol that fails both gates is not benefiting from communication at all.

---

## Output: provenance certificate

See `certificate-format.md` for the full schema. The certificate records:
- Seed set, condition definitions, and assumption verification status
- Per-seed Delta_content and Delta_structure values
- Sign counts and mean differences for both estimands
- Gate pass/fail status
- A plain-language summary suitable for an engineering manager

---

## Known limitations

- The permutation scheme assumes messages are exchangeable across agents at each timestep. Systems with directed communication graphs where message order is semantically meaningful may require a different scrambling scheme.
- The diagnostic cannot identify *which* component of the message content is doing useful work — only whether content as a whole adds value. Ablation over message dimensions requires a separate study.
- The diagnostic is designed for fixed, frozen policies. Dynamic or adaptive policies (including policies that condition on past messages) require careful matched-RNG design to avoid assumption violations.
