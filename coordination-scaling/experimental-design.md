# Experimental Design: Coordination Scaling

## Experiment 1 — Learned channel vs. sufficient statistic

**Phase 1 — Training:**
- CommNet: 3-dimensional continuous bottleneck; PPO outer loop; 5M environment steps; 5 runs (seeds 1001–1005)
- DIAL: 3-bit discrete bottleneck; same training protocol
- Task: SPS at α=0.06, M=4, N=256, 67 steps, dt=0.02

**Phase 2 — Representation analysis:**
- Run trained policy on 500 held-out trajectories (seeds 2001–2500)
- Compute Spearman ρ between each of the 3 bottleneck dimensions and 5 reference features: (v_x, v_y, f_valid, n_valid, local_particle_density)
- Cluster bottleneck activations by f_valid ∈ {0, (0, 0.33], (0.33, 0.67], (0.67, 1]} — checks whether validity is encoded even if not in a single dimension

**Phase 3 — Reward comparison:**
- Run trained CommNet and DIAL against `shared_summary_v2` and `capacity_matched_independent` on 16 matched seeds (seeds 3001–3016)
- Report mean yield gap: learned vs. hand-designed

**Pre-registration criterion:** At least one bottleneck dimension must achieve |ρ| > 0.4 with a reference feature for a representational comparison to be meaningful.

---

## Experiment 2 — √M team-size scaling

| M | K per agent | Shared channel |
|---|---|---|
| 1 | 8 | None |
| 2 | 4 | 3-slot v2 summary |
| 4 | 2 | 3-slot v2 summary |
| 8 | 1 | 3-slot v2 summary |

Fixed: α=0.06, N=256, dt=0.02, 67 steps. 16 matched seeds per M level.

**Analysis:**
1. Compute Δ̄(M) and 95% bootstrap CI per M level
2. Plot Δ̄(M) vs. √M with CI bands; overlay Proposition 2 theoretical prediction
3. Fit log-linear model: log Δ̄ ~ β log M; report β̂ with 95% CI; test H₀: β = 0.5
4. blend_w sensitivity at M=8: re-tune by line search on 4 pilot seeds; compare to frozen M=4 value

**Pre-registration gate:** Monotone positive relationship Δ̄(M=2) < Δ̄(M=4) < Δ̄(M=8) is the minimum for any scaling claim.
