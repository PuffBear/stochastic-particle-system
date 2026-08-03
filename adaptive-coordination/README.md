# Adaptive Coordination under Bounded Memory

**Research idea:** FR-B4
**Target venue:** ICLR 2028 (primary) · NeurIPS 2028 (fallback)
**Status:** Reproduction gate PASSED — full grid run1 complete (8 seeds/cell)

## Core question

SPS-C03 used a fixed field direction per episode. If the field rotates during an episode, agents face a trade-off: long memory gives low-variance field estimates but tracks a direction that may no longer exist; short memory tracks the current direction but with high variance.

This paper identifies the minimum memory length L — in steps — required to maintain positive coordination benefit as a function of field rotation speed ω. The key testable prediction: **L_critical(ω) ≈ c/ω** — the critical memory window scales as the field autocorrelation time. This follows directly from the information-theoretic argument that useful observations are those made within one autocorrelation time of the current step.

The (ω=0, L=all-steps) condition is exactly SPS-C03. Every result is measured as degradation relative to that confirmed baseline.

## Venue: ICLR 2028

Target submission deadline: ~October 2027. The 10–12 month timeline accommodates
the prerequisite engineering (now complete), the full 40-condition experiment grid,
analysis, and writing. ICLR is well-suited for the theoretical+empirical structure:
a falsifiable L_critical ~ 1/ω prediction grounded in information theory, tested
across multiple rotation speeds with matched counterfactuals.

## Engineering status

All Phase 2 prerequisite engineering is now implemented in the main SPS codebase:

1. **Rotating field** — `omega` parameter in `ParticleEnvConfig`; θ(t) sequence
   pre-generated at `reset()` as `theta_0 + omega * t * dt`; `step()` updates
   the orientation each step. At omega=0 this reduces to SPS-C03 exactly.

2. **Memory controllers** — `capacity_matched_velocity_controller_v2_window` and
   `capacity_matched_velocity_controller_v2_decay` in `policies.py`. Both accept
   a `history` list (oldest first) accumulated by the runner. At L=1 (current step
   only), the shared arm matches SPS-C03's per-step behaviour.

3. **Baseline adaptation** — both controllers apply the same field+density blend
   to both shared and independent arms; the only difference is the data source
   (team mean vs. self mean). This isolates the communication channel.

4. **Reproduction gate** — `analysis/run_fr_b4_adaptive_coordination.py` runs
   32 SPS-C03 confirmed seeds (6001–6032) at ω=0, L=1 before any non-zero ω runs.

## Files

- `research-questions.md` — field model, memory model, Q1–Q4, kill criteria
- `PLAN.md` — full publication plan including prerequisite engineering phases
- `theory/field-rotation.md` — rotating field model, L_critical derivation, theoretical predictions
- `experiments/grid-design.md` — pre-registered (ω, L, method) grid, seed plan, analysis
- `paper/outline.md` — ICLR 2028 section outline with draft abstract
