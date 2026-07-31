# Verified Literature Ledger

Last updated: 2026-07-31

Scope: primary research and official project sources nearest to the stochastic-particle collection benchmark. The search covered mobile particle collection, multi-robot source and flow-field search under local sensing, active-particle foraging, weak-signal detectability, irreversible aggregation, multi-agent benchmark infrastructure, and matched/common-random-number simulation design.

## Access-status legend

- **Full text read:** the paper's primary full text was inspected.
- **Abstract read:** only the publisher or arXiv abstract and bibliographic record were inspected.
- **Official code read:** the authors' or publisher-linked repository documentation was inspected.

## Closest task and benchmark precedents

| Source | Year | Primary or official URL | Read status | Exact relevance | Novelty threat | Baseline or design implication |
|---|---:|---|---|---|---|---|
| Wang, Ottino, Umbanhowar, and Lueptow, **“Mobile-collector capture of particles in a chaotic flow”** | 2025 | https://doi.org/10.1371/journal.pone.0329766 | Full text read | Studies one mobile circular collector capturing passive tracer particles in a two-dimensional double-gyre flow. The collector periodically measures its local particle distribution and selects motion using four strategies: fixed direction (FD), fixed angle relative to flow (FA), fixed target (FT), and advected moving target (MT). The paper analyzes trapping and capture-time scaling. Its conclusion explicitly names multiple collectors and coordination as future work. | **Critical nearest neighbor.** The present project cannot claim the first mobile particle collector, first locally particle-guided capture task, first vortex-flow collection study, or first capture-scaling analysis. | Implement faithful FD, FA, FT, and MT adaptations, or document why a strategy is inapplicable. Include a single-collector reproduction/analogue. Separate benefits from adding collectors from benefits caused by coordination. Treat this paper as the primary template and comparator. |
| Löffler, Panizon, and Bechinger, **“Collective foraging of active particles trained by reinforcement learning”** | 2023 | https://doi.org/10.1038/s41598-023-44268-3 | Full text read | Thirty locally observing active colloids use a shared PPO policy to forage randomly appearing food. Collective flocking and milling emerge despite individually defined rewards. | **High threat** to any claim of being the first local-sensing MARL particle-foraging or particle-collection system. | State the role reversal precisely: in this source, the particles are controlled agents and food is directly perceived; in the proposed benchmark, a small collector team acts while passive particles are environment entities and the transport signal must be inferred from particle motion. Include shared-policy PPO/IPPO and report whether collective behavior helps when direct target cues are absent. |
| Bhatt et al., **“Experimental Setup and Software Pipeline to Evaluate Optimization based Autonomous Multi-Robot Search Algorithms”** | 2025 | https://arxiv.org/abs/2506.16710 | Full text read | Presents an open physical testbed and software pipeline for four robots searching for a noisy acoustic source. Evaluates Bayes-Swarm, particle-swarm search, and random walk, and emphasizes signal-to-noise limitations. | **High threat** to a generic claim of introducing the first noisy multi-robot search benchmark or pipeline. | Random walk, swarm optimization, and a Bayesian informative-search method are established comparator families. The present benchmark must differentiate particle-motion inference, paired detectability measurement, capture, and aggregation from scalar-source localization. |
| Ghassemi, Balazon, and Chowdhury, **“A penalized batch-Bayesian approach to informative path planning for decentralized swarm robotic search”** | 2022 | https://doi.org/10.1007/s10514-022-10047-8 | Abstract read; official code read | Develops decentralized batch-Bayesian informative path planning for robots searching for a spatially varying signal, balancing exploration, exploitation, and inter-robot path overlap. | **Moderate threat** to claims that bounded communication or decentralized belief-guided search is itself novel. | Consider Bayes-Swarm-P only if a scientifically clean observation adapter can convert local particle-motion evidence into an inferred vector/scalar field. Do not force this baseline if that adapter changes the task. Official code: https://github.com/adamslab-ub/Bayes-Swarm-P |
| Atanasov, Le Ny, and Pappas, **“Distributed Algorithms for Stochastic Source Seeking With Mobile Robot Networks”** | 2015 | https://doi.org/10.1115/1.4027892 | Full paper read from author-hosted PDF | Provides model-free distributed stochastic-gradient control and model-based mutual-information-gradient control for teams seeking a noisy source. Handles imperfect formations and distributed computation. | **Moderate threat** to claims that cooperative noisy-field exploitation, distributed gradient estimation, or information sharing is new. | A distributed stochastic-gradient or consensus-flow controller is a mandatory related-work comparator when the planted field admits such a construction. At minimum, compare against local mean-flow and team-aggregated mean-flow estimators. |
| Gunnarson, Mandralis, Novati, Koumoutsakos, and Dabiri, **“Learning efficient navigation in vortical flow fields”** | 2021 | https://doi.org/10.1038/s41467-021-27015-y | Full text read | Shows that deep RL can navigate unsteady vortical flow from local measurements and that local velocity is substantially more useful than vorticity or no flow information in its task. | **Moderate threat** to claims that learning from local flow cues is new. | Include flow-blind and local-velocity policy ablations. The scripted local-flow estimator is indispensable; learned failure cannot establish task impossibility if local velocity scripting succeeds. |
| Patiño, Mayya, Calderon, Daniilidis, and Saldaña, **“Learning to Navigate in Turbulent Flows with Aerial Robot Swarms: A Cooperative Deep Reinforcement Learning Approach”** | 2023 | https://arxiv.org/abs/2306.04781 | Abstract read | Uses cooperative deep RL and nearest-neighbor information to help aerial robot teams compensate for turbulent flow and generalize to larger teams. | **Moderate AAMAS threat.** Information sharing for multi-robot flow navigation is established, although the task is target navigation rather than latent-signal detection and particle capture. | Compare independent local execution against a bounded neighbor/team summary. Demonstrate that any boundary shift is caused by shared evidence, not merely additional agents or centralized training. |
| Mecanna, Loisy, and Eloy, **“A critical assessment of reinforcement learning methods for microswimmer navigation in complex flows”** | 2025 | https://doi.org/10.1140/epje/s10189-025-00522-2 | Abstract and publisher preview read | Evaluates Q-learning, A2C, and PPO in partially observable flows and reports that PPO can greatly outperform simpler RL implementations when implemented and tuned carefully. | **Moderate methodological threat** to negative claims about RL algorithms based on a weak implementation. | Use a maintained PPO implementation, vectorized environments, recurrent policies where history is required, and a restrained but documented tuning budget. Do not interpret one failed learner as evidence that the benchmark signal is unusable. |
| Vergassola, Villermaux, and Shraiman, **“‘Infotaxis’ as a strategy for searching without gradients”** | 2007 | https://pubmed.ncbi.nlm.nih.gov/17251974/ | Abstract read | Introduces information-gain search for sparse, noisy source cues without usable gradients. | **Low-to-moderate threat** to generic claims about searching under weak and intermittent evidence. | Infotaxis is relevant only if the task exposes or constructs a belief over a source/field hypothesis. If included, describe the additional model assumptions. Otherwise cite it as a neighboring information-search family, not a mandatory direct baseline. |

## Multi-agent benchmark and learning precedents

| Source | Year | Primary or official URL | Read status | Exact relevance | Novelty threat | Baseline or design implication |
|---|---:|---|---|---|---|---|
| Bettini, Kortvelesy, Blumenkamp, and Prorok, **“VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning”** | 2022 | https://arxiv.org/abs/2207.03530 | Abstract read; official code read | Provides a vectorized PyTorch two-dimensional physics engine, customizable sensors and communication, and twelve multi-robot scenarios for MARL benchmarking. | **High infrastructure threat.** A custom continuous two-dimensional simulator is not by itself a contribution. | Either implement the environment as a VMAS scenario or justify a custom engine through passive-particle trajectory generation, event-keyed stochastic coupling, and attached-disc capture geometry. Benchmark throughput and determinism. Official code: https://github.com/proroklab/VectorizedMultiAgentSimulator |
| Lowe et al., **“Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments”** and the Multi-Agent Particle Environment | 2017 | https://arxiv.org/abs/1706.02275 | Abstract read; official environment repository read | Establishes a widely used simple continuous two-dimensional multi-agent particle world and centralized-training/decentralized-execution actor-critic baseline. | **Moderate infrastructure threat** to generic “particle-world benchmark” positioning. | Explain that the proposed particles are numerous non-agent stochastic entities, while collectors are the only agents. Do not describe the work as merely a new particle-world environment. Official environment: https://github.com/openai/multiagent-particle-envs |
| Yu et al., **“The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games”** | 2022 | https://papers.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf | Official paper and relevant benchmark description inspected; official code read | Establishes MAPPO and IPPO as strong, standard cooperative MARL baselines across multiple benchmark families. | **Baseline requirement**, not a direct novelty threat. | Include recurrent shared-parameter IPPO and one standard MAPPO implementation. Keep training budgets, observations, action spaces, and evaluation seeds aligned. Official MAPPO code: https://github.com/marlbenchmark/on-policy |
| de Witt et al., **“Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?”** | 2020 | https://arxiv.org/abs/2011.09533 | Abstract read | Shows that IPPO can match or exceed joint-learning methods in a major cooperative benchmark despite its theoretical limitations. | **Baseline requirement** for any claim that centralized critics or sharing are necessary. | IPPO is the correct learned independent-information comparator to MAPPO. A shared-summary environmental ablation must not be conflated with centralized training. |

## Detectability and physical sensing limits

| Source | Year | Primary or official URL | Read status | Exact relevance | Novelty threat | Baseline or design implication |
|---|---:|---|---|---|---|---|
| Lakhani and Elston, **“Testing the limits of gradient sensing”** | 2017 | https://doi.org/10.1371/journal.pcbi.1005386 | Full text read | Uses particle-based reaction-diffusion simulation to determine when shallow gradients can be resolved amid molecular noise and tests temporal averaging as a noise-reduction mechanism. | **High conceptual threat** to a broad claim that a signal-detectability boundary is novel. | Novelty must be the operational, policy-specific exploitation boundary under local particle-motion observations—not the existence of a physical sensing limit. Include temporal-history ablations and a direct signal-estimation diagnostic. |
| Mugler et al., **“Limits to the precision of gradient sensing with spatial communication and temporal integration”** | 2016 | https://doi.org/10.1073/pnas.1509597112 | Abstract read | Derives fundamental gradient-sensing precision limits for multicellular systems with communication and temporal integration. | **Moderate conceptual threat** to a generic “coordination improves weak-signal sensing” claim. | Interpret shared summaries as a finite communication channel and report how performance changes with collector count, history length, and message content. Avoid implying the qualitative benefit of aggregation across sensors is new. |
| Hein, Brumley, Carrara, Stocker, and Levin, **“Physical Limits on Bacterial Navigation in Dynamic Environments”** | 2016 | https://arxiv.org/abs/1512.04217 | Abstract read | Identifies dynamic regions in which temporal gradients can or cannot be resolved above noise and relates sensing limits to navigation. | **Moderate conceptual threat** to claims about a threshold separating resolvable and unresolvable transport cues. | Use a dimensionless, documented signal-to-noise axis rather than raw field amplitude alone. A useful per-step axis is \(\rho=\alpha\sqrt{\Delta t}/\sigma\); a length-scale alternative should be justified dimensionally. |

## Irreversible aggregation and capture mechanics

| Source | Year | Primary or official URL | Read status | Exact relevance | Novelty threat | Baseline or design implication |
|---|---:|---|---|---|---|---|
| Sidoravicius and Stauffer, **“Multi-particle diffusion limited aggregation”** | 2019 | https://doi.org/10.1007/s00222-019-00890-5 | Full text read | Studies an aggregate that irreversibly grows when diffusing particles contact its surface; attached particles become part of the capture geometry. | **Critical mechanics precedent.** Irreversible sticking and surface-expanding capture are established physical processes. | Do not claim attachment growth itself as novel. The defensible contribution is controlled decision-making by mobile collectors plus causal separation of true-field amplification from noise-triggered growth. Validate attached-disc geometry against simple DLA-style invariants. |
| Bressloff, **“Close encounters of the sticky kind: Brownian motion at absorbing boundaries”** | 2023 | https://doi.org/10.1103/PhysRevE.107.064121 | Abstract read | Provides encounter-based stochastic models for sticky and partially absorbing boundaries and relates contact time to absorption. | **Moderate mechanics threat** to treating contact/capture rules as unprecedented. | State clearly that the canonical benchmark uses immediate absorption at first geometric contact. Partial absorption or contact-time thresholds belong in robustness or future work. |
| Paoluzzi, Di Leonardo, and Angelani, **“Fractal aggregation of active particles”** | 2018 | https://doi.org/10.1103/PhysRevE.98.052603 | Abstract read | Studies aggregation and fractal structure in two-dimensional run-and-tumble active particles. | **Low-to-moderate mechanics threat** to broad active-matter aggregation claims. | Avoid claims about discovering fractal aggregation. Aggregate morphology may be a diagnostic, but the benchmark's primary outcome should remain interception and capture dynamics. |
| Teixeira et al., **“Collective ballistic motion explains fast aggregation in adhesive active matter”** | 2026 | https://arxiv.org/abs/2508.11793 | Full-text preprint inspected | Identifies rapid aggregation regimes in self-aligning adhesive active particles and explains a flocking-mediated ballistic transition. | **Low-to-moderate conceptual threat** to broad “aggregation causes cascades” language. | Use “capture cascade” as an operational benchmark statistic, not as a claim of a new aggregation phase. Distinguish passive-particle attachment to controlled collectors from active inter-particle adhesion. |

## Matched simulation and common-random-number design

| Source | Year | Primary or official URL | Read status | Exact relevance | Novelty threat | Baseline or design implication |
|---|---:|---|---|---|---|---|
| Glasserman and Yao, **“Some Guidelines and Guarantees for Common Random Numbers”** | 1992 | https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf | Full paper read | Formalizes common random numbers (CRN) as a coupling intended to induce positive outcome covariance and reduce the variance of simulation comparisons. Discusses synchronization and conditions under which CRN is beneficial. | **Methodological precedent.** Matched stochastic episodes are not novel on their own. | Analyze paired differences directly and report the observed paired covariance and variance reduction. Keep distinct random streams or keys for each stochastic input. |
| Wright and Ramsay, **“On the Effectiveness of Common Random Numbers”** | 1979 | https://doi.org/10.1287/mnsc.25.7.649 | Abstract read | Demonstrates a simulation setting in which CRN induces negative response correlation and increases rather than decreases variance. | **Validity threat** to assuming matching automatically improves inference. | Predefine a fallback to independent-replicate uncertainty if empirical coupling is harmful. Report both paired and unpaired standard errors during the pilot. |
| Buffalo, Pearson, and Klein, **“Realizing Common Random Numbers: Event-Keyed Hashing for Causally Valid Stochastic Models”** | 2026 | https://arxiv.org/abs/2603.11084 | Full text read | Shows that reusing one stateful PRNG seed can misalign downstream draws when an intervention changes program control flow. Proposes counter-based, event-keyed random numbers so the same modeled event receives the same exogenous noise across counterfactual runs. | **Critical implementation threat.** A same-seed signal/null pair is not necessarily a scientifically valid matched pair. Capture and growing geometry can change control flow immediately. | Noise must be indexed by stable event identity, such as `(pair_id, particle_id, timestep, component)`, using a counter-based generator, or pre-generated as a complete tensor before either episode begins. Noise must be generated for all particle-time keys even after a particle is captured. Add a test proving equality of corresponding null/signal noise tensors. |

## Defensible novelty after this search

The literature does **not** support claims that this is the first:

- mobile collector of passive particles;
- local particle-distribution-guided capture strategy;
- multi-agent or active-particle foraging task;
- multi-robot search problem under noisy local sensing;
- RL navigation method using local flow information;
- irreversible particle-attachment process;
- signal-detectability threshold; or
- matched stochastic simulation design.

A defensible contribution remains:

1. a controlled **multi-collector** benchmark in which collectors infer a weak latent transport field from local histories of passive-particle motion;
2. a **policy-specific signal-exploitation boundary** estimated from event-keyed paired signal/null episodes;
3. an isolated test of whether **bounded shared summaries shift that boundary** beyond the gain from simply deploying more independent collectors;
4. a decomposition of **first interception** from **post-contact irreversible growth**, including null-triggered false cascades; and
5. a released paired trajectory dataset supporting counterfactual analysis of evidence, interception, and growth.

Until a broader systematic search is complete, the manuscript should use “we introduce” rather than an absolute “first.”

## Recommended primary research question

The current phrase “locally observing collectors” does not define a unique boundary: the boundary depends on the policy, field family, outcome, horizon, and inferential criterion.

A narrower AAMAS-defensible question is:

> Under canonical uniform drift, does sharing only per-agent local mean-velocity summaries lower the smallest per-step signal-to-noise ratio \(\rho=\alpha\sqrt{\Delta t}/\sigma\) at which four otherwise identical local-flow collectors achieve a prespecified positive paired gain \(G=\min(T_0,H)-\min(T_\alpha,H)\) in horizon-censored time to first capture, compared with independent collectors?

This is one coordination question. Vortex generalization, learning baselines, scaling, and growing geometry are validation or mechanism analyses rather than additional clauses in the research question.

## Mandatory baseline stack

1. Random movement.
2. Systematic coverage.
3. Density-greedy collection.
4. Wang et al. FD, FA, FT, and MT adaptations.
5. Scripted local-flow following.
6. Scripted shared/team-flow following.
7. One collector, four independent collectors, and four shared-summary collectors.
8. Flow-blind recurrent IPPO.
9. Local-velocity recurrent IPPO.
10. Standard MAPPO.
11. Privileged true-field oracle.
12. Bayes-Swarm-P or distributed stochastic-gradient control only if the observation adapter is scientifically faithful and documented.

## Immediate scientific gates

- Treat Wang et al. (2025) as the nearest neighbor in the introduction, baselines, and reviewer audit.
- Use event-keyed or pre-generated particle noise; a reused base seed is insufficient.
- Estimate a separate boundary for every policy and metric rather than claiming a universal environment threshold.
- Demonstrate a coordination effect beyond independent parallel search; otherwise the work is more naturally a flow/robotics benchmark than an AAMAS paper.
- Keep irreversible growth out of the primary research question unless the project is explicitly split into a second paper.
