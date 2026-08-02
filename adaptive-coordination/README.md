# Adaptive Coordination under Bounded Memory

**Research idea:** FR-B4
**Target venue:** IJCAI 2027 · ICLR 2027
**Status:** 2–3 months out; requires stable fixed-field SPS baseline

## Core question

The SPS-C03 result used a fixed field direction per episode. If the field rotates during an episode, agents face a trade-off between tracking accuracy (recent observations, high variance) and historical breadth (cumulative history, possibly stale).

This paper identifies the minimum memory length L — in steps — required to maintain positive coordination benefit as a function of field rotation speed ω. The key theoretical prediction: L_critical(ω) scales as 1/ω (the field autocorrelation time). This is a concrete, testable claim tied to an information-theoretic argument, and the ground-truth field direction is known at every step, making the comparison exact.

## Files

- `research-questions.md` — field model, memory model, sub-questions, estimands, kill criteria
