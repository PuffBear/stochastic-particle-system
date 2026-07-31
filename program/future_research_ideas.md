# Future Research Ideas

Inactive directions only. None may consume active-paper resources without a Program Director decision.
Each entry is grounded in evidence from this codebase and targets an A/A* venue.

---

## SPS-FR-A — Communication as an escape from the dilution trap

- **Provenance:** SPS-P02 `baseline_summary.json`; SPS-WO-05 yield gate; single-collector diagnostic row in Table 2 of manuscript.
- **Motivating evidence:** At ρ=2, a single collector achieved mean gain 0.06938 while four independent collectors achieved 0.01104 — worse by a factor of 6. Area coverage alone does not explain this: four collectors cover more arena, not less. The gap suggests independent parallel search actively degrades each collector's signal-to-noise ratio by fragmenting the local particle sample each agent can average over.
- **Research question:** Does bounded team communication recover single-agent estimation efficiency as team size M grows, and is there a critical communication budget below which the dilution pathology persists regardless of policy?
- **Falsifiable hypothesis:** At fixed total sensing budget (M × K particles visible across the team), the matched yield gap between M=4 shared and M=4 independent closes monotonically with communication budget B, and at B=0 the gap is negative (independent is worse than M=1 matched-budget single agent).
- **Possible contribution:** Communication not as a coordination add-on but as a correction to a fundamental pathology of decentralised sensing. A phase-transition characterisation of the minimum communication budget needed to escape dilution.
- **Nearest literature / novelty risk:** Team-size effects in multi-robot search (Burgard et al. 2005) and multi-agent coverage (Schwager et al. 2011) study area allocation but not signal-to-noise dilution from observation fragmentation. The dilution framing is novel if the matched-budget control holds.
- **Minimal experiment:** Run M ∈ {1, 2, 4, 8} under matched total sensing budget (constant M×K); compare independent vs. shared-summary yield at each M; test whether shared-minus-independent grows with M.
- **Kill criterion:** The single-collector advantage disappears under passive hazard matching, or communication benefit is flat across M.
- **Candidate venue:** AAMAS 2027 (abstract 1 October 2026, paper 8 October 2026).
- **Infrastructure reuse:** High — same environment, same CE metrics, same runner. Requires adding M-parameterised episode configs.
- **Status / priority:** `unvalidated`, 9/10.

---

## SPS-FR-B — Decentralised Bayesian field estimation under bounded communication

- **Provenance:** Observation contract in `src/particle_benchmark/observations.py`; apparent-velocity validity logic; the 3-number summary design in SPS-WO-04.
- **Motivating evidence:** Each collector implicitly maintains a local estimate of the field direction b̂ from causally valid apparent particle velocities. The team-mean summary (v̄_x, v̄_y, f_valid) is one aggregation of those local estimates. Whether it is optimal under a 3-scalar budget is an open question with a tractable theoretical answer.
- **Research question:** What is the minimum-variance unbiased estimator of b̂ under a budget of B shared scalars, and how much information does the team-mean summary leave on the table relative to sharing posterior mean and variance (5 scalars)?
- **Falsifiable hypothesis:** Under a 5-scalar budget (posterior mean + variance per agent), the coordinated team achieves a strictly lower mean squared error in field estimation than under the 3-scalar team-mean, and this gap predicts CE rank across architectures.
- **Possible contribution:** A closed-form characterisation of the Cramér-Rao bound for decentralised field estimation under communication constraints; an empirical test of whether CE correlates with proximity to the bound.
- **Nearest literature / novelty risk:** Distributed estimation (Ribeiro & Giannakis 2006), consensus-based filtering (Olfati-Saber 2007). Novelty requires explicit communication-budget parameterisation tied to a physical MARL task, not just an estimation theory result.
- **Minimal experiment:** Compute empirical MSE(b̂) for shared, independent, and oracle policies across seeds; fit against the theoretical bound; test whether CE predicts estimation accuracy rank.
- **Kill criterion:** The team-mean achieves the bound at B=3, leaving no room for the 5-scalar version, or MSE does not predict CE.
- **Candidate venue:** IJCAI 2027 or NeurIPS 2026 (theory track).
- **Infrastructure reuse:** Medium — needs posterior tracking added to the observation pipeline; empirical validation reuses existing rollouts.
- **Status / priority:** `unvalidated`, 8/10.

---

## SPS-FR-C — Pursuit-evasion with latent field and asymmetric information

- **Provenance:** FR-005 (passive-particle limitation); AAMAS relevance audit in manuscript Section 8.
- **Motivating evidence:** The current environment treats particles as passive. The pursuit-evasion extension makes them adversarial: each particle observes collectors within radius r_e and responds to minimise capture probability. The latent field gives collectors a structural advantage (they can anticipate particle drift) but evaders can exploit the same field (drift-assisted evasion in favourable directions). This is a two-team zero-sum game with shared latent state.
- **Research question:** At a fixed latent field strength α, what is the Nash equilibrium capture rate, and how does the collector team's communication budget shift the equilibrium?
- **Falsifiable hypothesis:** Collector teams with bounded shared summaries achieve a higher Nash equilibrium capture rate than independent collectors, and the gap grows with α up to a catchability ceiling.
- **Possible contribution:** A genuine pursuit-evasion game with latent environment state, asymmetric information (collectors observe particle positions; evaders observe collector positions), and a natural role for communication in breaking informational symmetry.
- **Nearest literature / novelty risk:** Pursuit-evasion is well-studied (Isaacs 1965; Vidal et al. 2002). The novelty must come from the latent field as a shared exploitable structure — neither side observes it directly but both can infer it.
- **Minimal experiment:** Add a reactive evader policy (move away from nearest collector, weighted by distance); measure equilibrium capture rate under shared vs. independent collector policies; compare to passive-particle baseline.
- **Kill criterion:** Equilibrium capture rate is indistinguishable between shared and independent collectors, or the evasion response trivially dominates regardless of communication.
- **Candidate venue:** AAMAS 2027 main track.
- **Infrastructure reuse:** Medium — needs adversarial particle policy added to `dynamics/particles.py`; everything else reuses existing infrastructure.
- **Status / priority:** `unvalidated`, 8/10.

---

## SPS-FR-D — Shapley decomposition of team communication value

- **Provenance:** SPS-WO-05 per-collector capture counts in `episode_summaries.jsonl`; multi-agent fairness literature.
- **Motivating evidence:** Per-collector capture counts are already recorded in every episode summary. With M=4 collectors, Shapley values can be computed from existing rollout data by enumerating subsets of agents and measuring marginal yield contribution. With and without the shared summary, the Shapley allocation changes — this change is itself a measurable coordination effect.
- **Research question:** Does team communication symmetrise Shapley values (making all agents equally valuable) or create a hierarchy (one agent's information dominates)? And does symmetrisation correlate with higher team yield?
- **Falsifiable hypothesis:** Under the shared-summary policy, the Gini coefficient of Shapley values across agents is strictly lower than under the independent policy, and this reduction predicts CE across seeds.
- **Possible contribution:** Communication value decomposed through cooperative game theory — a result connecting coordination efficiency to intra-team fairness. Shapley values as a diagnostic for whether communication homogenises or specialises agent roles.
- **Nearest literature / novelty risk:** Shapley value credit assignment in MARL (Wang et al. 2020 SHAPLEY-Q). Novelty requires the physical interpretation — communication as a Shapley symmetriser — and the empirical correlation with CE.
- **Minimal experiment:** Compute leave-one-out marginal yields from existing SPS-WO-05 episodes; estimate Shapley values by sampling subsets; compare Gini coefficient under shared vs. independent; test correlation with CE_s.
- **Kill criterion:** Shapley values are already symmetric under independent policy (all agents interchangeable), or the Gini change does not correlate with yield improvement.
- **Candidate venue:** AAMAS 2027 or JAAMAS (extended version).
- **Infrastructure reuse:** Very high — computable from existing episode logs without new runs.
- **Status / priority:** `unvalidated`, 8/10.

---

## SPS-FR-E — Curriculum learning across field strengths for sparse-reward MARL

- **Provenance:** MARL baseline training infrastructure in `src/particle_benchmark/marl/`; sparse reward diagnosis from trainer design discussion; CE framework in `src/particle_benchmark/metrics/coordination.py`.
- **Motivating evidence:** Training directly at α=0.06 produces a very sparse reward signal — captures are rare early in training, gradients are weak, and all six MARL architectures start from near-zero performance. A curriculum from α=1.0 (strong field, easy to detect, dense reward) down to α=0.06 might produce policies with better CE scores at the target regime, and the matched counterfactual framework can test generalisation cleanly across the full α spectrum.
- **Research question:** Does curriculum training across field strengths produce policies that achieve higher CE on held-out α values than policies trained directly at the target, and is there a curriculum schedule that minimises the gap between training performance and transfer performance?
- **Falsifiable hypothesis:** A monotone-decreasing α curriculum achieves strictly higher CE at α=0.06 than direct training at α=0.06, measured on matched evaluation seeds the curriculum never saw.
- **Possible contribution:** A principled curriculum design framework for latent-field environments, with a theoretical connection between field strength and reward density that motivates the schedule. Broader lesson: matched counterfactual evaluation as a curriculum design signal.
- **Nearest literature / novelty risk:** Curriculum RL (Bengio et al. 2009; Portelas et al. 2020). The novelty requires the physical justification (field strength → reward density → curriculum signal) and the counterfactual evaluation protocol, not just applying curriculum RL to this environment.
- **Minimal experiment:** Train IPPO and CommNet under three schedules (direct α=0.06; step curriculum α∈{1.0, 0.5, 0.25, 0.06}; continuous annealing); evaluate CE on 8 held-out seeds at α=0.06; compare.
- **Kill criterion:** Direct training matches curriculum performance, or transfer CE does not exceed within-distribution CE.
- **Candidate venue:** NeurIPS 2026 or ICLR 2027 (generalisation / curriculum track).
- **Infrastructure reuse:** High — same MARL training loop, same CE metrics, same environment; needs multi-α training wrapper added to trainer.py.
- **Status / priority:** `unvalidated`, 7/10.

---

## SPS-FR-F — Heterogeneous sensing teams and emergent role specialisation

- **Provenance:** FR-006 (homogeneous-team limitation); observation budget K=32 in `src/particle_benchmark/observations.py`; team-mean summary design in SPS-WO-04.
- **Motivating evidence:** All four collectors currently share the same sensing budget K=32 nearest particles. Under a fixed total budget of 4×32=128 particle slots, a heterogeneous team might allocate asymmetrically — one high-bandwidth collector (K=96) serving as a field estimator whose summary the other three (K=16 each) use. If agents learn to reproduce the team-mean summary structure autonomously, it validates the SPS-C03 probe as a natural equilibrium, not an arbitrary design choice.
- **Research question:** Under fixed total sensing budget M×K, do heterogeneous sensing teams spontaneously specialise — with the high-bandwidth agent's local estimate dominating the shared summary — and does this match the CE achieved by the hand-coded team-mean?
- **Falsifiable hypothesis:** A MAPPO or CommNet team trained with heterogeneous sensing budgets (one K=96, three K=16) achieves CE within 0.1 of the matched-budget homogeneous team, and the high-K agent's message correlation with b̂ is strictly higher than the low-K agents'.
- **Possible contribution:** Role emergence as a response to sensing asymmetry; validation that the fixed-statistic probe in SPS-C03 is a natural communication equilibrium under constrained sensing.
- **Nearest literature / novelty risk:** Role emergence in MARL (Wang et al. 2021 ROMA; Christianos et al. 2021). Novelty requires the physical sensing-budget constraint and the connection back to the fixed-statistic probe — not just demonstrating role emergence in a cooperative game.
- **Minimal experiment:** Train MAPPO under K∈{(32,32,32,32), (96,16,16,16), (64,32,16,16)} at fixed total budget; measure per-agent message correlation with b̂; compare CE across configurations.
- **Kill criterion:** CE is flat across sensing distributions, or the high-K agent does not become the dominant field estimator.
- **Candidate venue:** AAMAS 2027 main track.
- **Infrastructure reuse:** Medium — needs sensing budget parameterisation added to `ParticleEnvConfig` and `observations.py`; training and evaluation infrastructure reuses existing code.
- **Status / priority:** `unvalidated`, 7/10.
