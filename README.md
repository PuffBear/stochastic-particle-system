# Future Ideas

A working repository for research and industry directions built on the Stochastic Particle System (SPS) programme. The SPS paper confirmed that a 3-number shared team signal (mean velocity x, mean velocity y, fraction of valid observations) produces a statistically significant coordination gain over capacity-matched independent agents at α=0.06.

## Confirmed baseline (SPS-C03, 2026-08-01)

| Metric | Value |
|---|---|
| Mean coordination gain | +1.19 unique particles |
| 95 % bootstrap lower bound | +0.459 |
| Seeds positive | 20 / 32 |
| Shared channel size | 3 numbers (v_x, v_y, f_valid) |
| Condition | α=0.06, M=4, N=256, 67 steps, dt=0.02 |
| Attribution | Bandwidth structure drives most gain; count-weighted content adds ~+1.0 particle (6/8 seeds) |

MARL baselines (6 architectures × 8 seeds) are actively running on ShARC HPC (WO-08).

## Research directions

| Folder | Research ideas covered | Target venue |
|---|---|---|
| `communication-tradeoffs/` | FR-A1 + FR-A2 — failure boundary & bandwidth curve | AAMAS / ICML 2027 |
| `coordination-scaling/` | FR-B1 + FR-B2 — learned channels & √M scaling | ICLR / NeurIPS 2027 |
| `catchability-benchmark/` | FR-B3 — nondimensional two-axis benchmark | ICML 2027 |
| `adaptive-coordination/` | FR-B4 — nonstationary fields, bounded memory | IJCAI / ICLR 2027 |
| `offline-marl-evaluation/` | FR-C1 — offline policy ranking from paired datasets | NeurIPS 2027 |
| `strategic-heterogeneous-teams/` | FR-C2 + FR-C3 — heterogeneous agents & strategic evaders | AAMAS 2027 |

## Industry directions

| Folder | Concept |
|---|---|
| `coordination-diagnostic/` | Tool: diagnose whether a team's communication protocol is helping or hurting |
| `environmental-collection/` | Decision layer for mobile vs. stationary collection planning |
| `simulation-audit/` | V&V audit for event-driven simulation conclusions |

## Source files

- `future_research_ideas.md` — canonical research pipeline (all tiers, status, kill criteria)
- `future_industry_ideas.md` — canonical industry pipeline (unit economics, validation gates)

## Sequencing note

None of the industry ideas should be commercialized before the SPS paper is submitted. The paper is the credibility artifact that makes all three fundable. FR-A3 (ablation completeness) is not in this branch — it is a mandatory component of the current SPS submission, not a future paper.
