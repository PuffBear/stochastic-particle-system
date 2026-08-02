# Adaptive Coordination under Bounded Memory

**Research idea:** FR-B4
**Target venue:** NeurIPS 2027 (primary) · ICLR 2028 (fallback)
**Status:** 4–6 weeks of prerequisite engineering before experiments can run

## Core question

SPS-C03 used a fixed field direction per episode. If the field rotates during an episode, agents face a trade-off: long memory gives low-variance field estimates but tracks a direction that may no longer exist; short memory tracks the current direction but with high variance.

This paper identifies the minimum memory length L — in steps — required to maintain positive coordination benefit as a function of field rotation speed ω. The key testable prediction: **L_critical(ω) ≈ c/ω** — the critical memory window scales as the field autocorrelation time. This follows directly from the information-theoretic argument that useful observations are those made within one autocorrelation time of the current step.

The (ω=0, L=all-steps) condition is exactly SPS-C03. Every result is measured as degradation relative to that confirmed baseline.

## Why NeurIPS 2027, not ICLR 2027

ICLR 2027 deadline is ~October 2026. FR-B4 requires a rotating-field environment that doesn't yet exist, two new controller variants (sliding window, exponential decay), and a new theory derivation. 4–6 weeks of prerequisite engineering before experiments start; then 4–6 weeks of experiments and analysis; then 8–10 weeks of writing. NeurIPS 2027 (~February 2027 deadline) is the earliest realistic target. ICLR 2028 is the fallback.

## Prerequisite engineering (before any experiment runs)

1. Rotating field: θ(t+dt) = θ(t) + ω·dt with pre-generated θ(t) sequences for matched counterfactuals
2. Memory controllers: sliding window (last L steps) and exponential decay (λ = exp(−1/L)) variants of shared_summary_v2
3. Baseline adaptation: capacity_matched_independent must use the same memory model as the shared arm
4. ω=0 reproduction gate: modified environment must reproduce SPS-C03 result before experiments proceed

## Files

- `research-questions.md` — field model, memory model, Q1–Q4, kill criteria
- `PLAN.md` — full publication plan including prerequisite engineering phases
- `theory/field-rotation.md` — rotating field model, L_critical derivation, theoretical predictions
- `experiments/grid-design.md` — pre-registered (ω, L, method) grid, seed plan, analysis
- `paper/outline.md` — full NeurIPS section outline with draft abstract
