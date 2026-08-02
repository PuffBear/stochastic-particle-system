# Research Questions: Strategic and Heterogeneous Teams

## Primary research question

When agents have heterogeneous sensing quality (α_i ≠ α_j) or particles have bounded evasion capacity ε > 0, does the communication structure that produced the SPS-C03 coordination gain remain beneficial, become suboptimal, or become harmful — and does the direction of change match the theoretical prediction in each case?

## Sub-questions

### Q1 — Heterogeneous sensing: does Proposition 2 extend?

**Theoretical prediction:** For M agents with sensing qualities {α_i}, the optimal team message weights agent i's contribution by α_i² (inverse variance). A team with one strong sensor (α_1=0.12) and three weak ones (α_2=α_3=α_4=0.03) should weight the strong sensor ~16× more than any individual weak sensor.

**Hypothesis:** A heterogeneity-aware controller (`heterogeneous_summary_v3`, using α_i²-weighted mean) outperforms the homogeneous v2 controller by at least +0.5 particles at the asymmetric condition across 8 matched seeds. The homogeneous v2 over-weights weak agents and under-weights the strong one — violating the inverse-variance principle.

**Condition:** M=4, one strong agent (α=0.12) + three weak (α=0.03). Ensemble mean α=0.06 matches the C03 condition so absolute task difficulty is comparable.

**Arms:**

| Controller | Rule |
|---|---|
| `shared_summary_v2` | Count-weighted mean — ignores quality heterogeneity |
| `heterogeneous_summary_v3` | α_i²-weighted mean (Proposition 2 extension) |
| `capacity_matched_independent` | No sharing — lower bound |
| `oracle_routing` | Routes full local obs of strong agent to all — upper bound |

**Estimand:** E[Y(v3) − Y(v2)] and E[Y(v3) − Y(independent)] on 8 matched seeds.

### Q2 — Heterogeneous actuation: does specialization emerge?

**Setup:** One fast agent (v_max=0.5) + three slow (v_max=0.15). Equal sensing quality α=0.06 for all.

**Hypothesis:** A routing rule that delivers field estimates preferentially to the fast agent (who can exploit them) outperforms equal delivery across all agents.

**Estimand:** Compare `shared_summary_v2` (equal delivery) vs. `actuation_specialized_v1` (field estimate routed to fast agent) on 8 seeds.

### Q3 — Strategic evaders: does sharing help or hurt?

**Evader model:**
- Each particle has evasion budget ε (max distance per step in evasion direction)
- At each step, if any collector is within observation radius r_obs=0.15, particle moves ε away from the nearest collector
- Outside r_obs, particle moves as standard Brownian with field bias
- Particles observe only collector positions, not the communication channel

**Hypothesis:** At ε < 0.03, evasion is too weak to change collective behavior — C03 coordination benefit persists. At ε ≥ 0.08, strategic evasion exploits the implicit position concentration that team communication creates: a communicating team converges spatially, letting particles predict and avoid the cluster. Independent agents' spatial diversity makes escape harder.

**Estimand:** E[Y(shared) − Y(independent)] as a function of ε ∈ {0, 0.02, 0.04, 0.08, 0.15}. Does the sign of the contrast flip at some ε_critical?

### Q4 — Robustness of communication under evasion

**Hypothesis:** A modified communication structure that adds calibrated noise to the team's shared direction reduces position disclosure and restores positive gain against strategic evaders at ε=0.08.

**Estimand:** E[Y(noisy_shared_ε) − Y(shared_v2)] at ε=0.08. Report the noise level that maximizes Δ̄ against strategic particles.

## Kill criteria

- **Q1 kill:** Heterogeneity-aware v3 achieves the same gain as homogeneous v2 — quality heterogeneity at M=4 is empirically indistinguishable in the yield metric. Paper reduces to a theory-only contribution.
- **Q3 kill:** Coordination benefit persists at ε=0.15 — strategic evasion is too weak at any tested budget to eliminate the gain. The game-theoretic tension is absent; the paper reports only robustness.
