# Fresh AAMAS Main-Track Review

**Manuscript:** *Detecting Weak Structure in Stochastic Particle Systems: A Matched-Counterfactual Multi-Collector Benchmark*  
**Review date:** 2026-07-31  
**Package reviewed:** the frozen manuscript, project specification, claim and experiment ledgers, literature ledger, source code, tests, configurations, schemas, and dataset contract supplied with the submission.  
**Independence statement:** I did not use any previous review, rebuttal, target score, desired outcome, internal decision log, report, work order, handoff, or future-work document.

## Summary

This submission proposes a continuous two-dimensional benchmark in which four mobile collectors use local observations of many stochastic passive particles to exploit a weak latent transport field. Its intended primary outcome is a policy-relative, grid-censored signal boundary: the smallest preregistered drift-to-Brownian ratio for which a frozen local policy obtains a positive matched reduction in time to first interception relative to a no-signal episode. Signal and null episodes are intended to share initial state, Brownian forcing, field nuisance variables, policy randomness, and tie-breaking provenance. The paper also proposes to distinguish pre-contact search from post-contact irreversible aggregation, and eventually to compare independent collectors, bounded sharing, IPPO, MAPPO, and privileged controls.

The supplied implementation is a meaningful correctness-first simulator slice. It includes reflected particle and collector dynamics, deterministic capture-free resets, fixed and provisional growing capture, local observations with causal velocity-validity masks, stationary/random/privileged smoke policies, primary metric helpers, and closed top-level trajectory and run-manifest schemas. I independently ran the supplied test command and observed all 44 tests pass.

However, the submission explicitly contains no scientific experiment, no generated dataset, no frozen `local_flow_v1` policy, no complete matched environment-pair runner, no boundary estimator, and no multi-agent coordination result. The manuscript also identifies event-keyed tie resolution as a blocker. Thus the empirical contribution advertised by the title and abstract is not yet instantiated. As a main-track research paper, this is currently an unusually careful design document and software skeleton, not a completed benchmark study.

## Strengths

1. **Honest and disciplined claim boundary.** The manuscript clearly distinguishes implemented engineering invariants from scientific evidence and does not manufacture performance claims. All three ledger claims remain marked `proposed`.
2. **Good causal comparison instinct.** Pre-generating Brownian forcing and separating named random streams are appropriate foundations for paired counterfactual comparisons. The manuscript correctly recognizes that a reused stateful random stream can become invalid after trajectories diverge.
3. **Useful event-stage decomposition.** Separating time to first interception from post-contact aggregation is well motivated because growing capture geometry can amplify a chance initial contact.
4. **Clear primary sampling unit and estimand.** Treating the scenario seed—not particles, collectors, or timesteps—as the independent unit avoids a common pseudoreplication error. The sign and censoring convention for the paired first-interception metric are explicit.
5. **Substantial correctness work for this stage.** The supplied 44-test suite covers deterministic reset, action bounds, reflection, random-stream isolation, zero-signal rollout identity, visibility masking, ownership, event timing, smoke policies, and schema structure. The code is small enough to audit and the manuscript exposes remaining blockers rather than hiding them.
6. **Literature ledger is unusually threat-oriented.** It identifies close mobile-collector, local-sensing foraging, multi-robot search, flow navigation, aggregation, simulator, MARL, sensing-limit, and common-random-number precedents, and narrows novelty accordingly.

## Major weaknesses

### 1. The central empirical contribution does not yet exist

The paper asks for the weakest signal strength exploited by `local_flow_v1`, but that policy has not been implemented or frozen. There are no pilot trajectories, no signal grid with a seed budget, no estimate of the paired response curve, no simultaneous confidence-bound procedure, and no boundary result. Even the artifact writer and analysis script are absent. The protocol marks the performance run as unauthorized. This is fatal for a paper whose contribution is an empirical measurement instrument and benchmark result.

Unit tests establish that selected code paths behave as intended; they cannot establish that the task is nontrivial, that the signal is behaviorally usable, that the estimator is calibrated, or that the proposed boundary is scientifically meaningful.

### 2. The multi-agent contribution is not isolated

The active question concerns a “team” of four collectors, but it does not test a specifically multi-agent claim. Four identical collectors can simply provide four times the search opportunity. The current implemented observations expose teammate positions, yet no coordination mechanism, shared evidence channel, complementary role, congestion cost, collision constraint, or team-vs-independent comparison is implemented. MAPPO and IPPO are only names in an evaluation plan.

As written, the central result could become “one locally informed mobile-collector task replicated four times.” That would fit stochastic control, active matter, or robotics more naturally than the AAMAS main track. Claim SPS-C03—a bounded shared summary lowering the boundary—is much more directly multi-agent, but it remains secondary and wholly unevaluated.

### 3. Important causal-validity blockers remain in the executable system

The manuscript correctly notes that capture ties use a stateful PRNG. Once signal and null trajectories differ, different tie events can consume different draws, breaking event-level counterfactual alignment. There is also no wrapper that guarantees matched policy-randomness and complete episode provenance under adaptive policies.

More importantly, capture is detected only from positions at the end of each discrete step. A particle and collector can cross within a timestep without their endpoints lying within the capture radius. With the canonical parameters, Brownian displacement is not obviously negligible relative to the capture radius. Missed segment contacts, reflected subpaths, and timestep choice can therefore move the first-contact distribution—the primary outcome. A timestep-convergence analysis or continuous segment/contact rule is required.

The observation called “physical finite-difference particle velocity” is computed from reflected endpoint displacement. Near a reflecting wall this need not equal the physical path velocity or pre-reflection displacement. The paper should define it as apparent endpoint velocity or reconstruct the reflected path consistently.

### 4. The statistical boundary protocol is underspecified

The manuscript states a one-sided simultaneous 95% lower confidence bound but does not specify how it will be constructed, how many grid points it covers, or how the seed budget will be determined. The confirmatory signal grid itself is not present in the primary protocol. The pilot configuration lists amplitudes, but the registered experiment does not identify those values as the frozen confirmatory grid.

The term “boundary” also suggests a stable or monotone transition. The proposed rule selects the smallest significant tested point even if stronger points fail or the response is non-monotonic. Uniform drift plus reflection, policy dynamics, and finite horizon can readily produce non-monotone effects. The full response curve must be reported, and the paper must either justify monotonicity, adopt a persistence criterion, or explicitly call the output a first grid crossing rather than a boundary.

No variance, power, multiplicity, estimator-calibration, paired-covariance, or harmful-coupling diagnostic exists yet. The literature ledger itself notes that common random numbers can increase variance, but the protocol does not yet freeze the fallback or sensitivity analysis.

### 5. Mundane explanations are planned but not ruled out

A positive signal/null difference would not by itself establish weak-signal inference or coordination. Plausible alternatives include:

- passive flux or signal-induced boundary accumulation;
- increased swept area from having four collectors;
- lucky interception amplified by finite horizon or capture geometry;
- deterministic collector lattice interacting with field orientation;
- direct use of an externally aligned Cartesian frame rather than inference from local evidence;
- a policy following current local density rather than estimating transport;
- missed within-step contacts or timestep-dependent reflection artifacts;
- richer information, larger capacity, or a larger action budget in the shared or learned condition;
- tuning and evaluation on the same scenario seeds;
- an isolated significant grid point rather than a stable transition.

Stationary, random, coverage, density, velocity-shuffle/history-shuffle, one-collector, independent-team, shared-summary, and privileged controls are therefore not optional presentation details; they determine the interpretation.

### 6. Related work in the manuscript is far too thin

The internal literature ledger is broad and appropriately skeptical, but the manuscript cites only two sources. It omits the common-random-number methodology supporting its core design, event-keyed causal coupling, sensing-limit precedents, multi-agent simulator precedents, IPPO/MAPPO references, distributed noisy search, and aggregation mechanics. The paper consequently does not yet demonstrate its novelty boundary to an AAMAS reader. A nearest-neighbor comparison table would be more effective than an expansive “we introduce” narrative.

### 7. Reproducibility is promising but incomplete

The package provides source, tests, configs, and schemas, but there is no environment lockfile, CI record, command-line runner, config parser tying YAML values to the executable environment, trajectory writer, schema-instance validator, checksum pipeline, raw artifact, or analysis script. The schema tests only inspect selected schema keys; they do not validate valid and invalid example records. The manifest permits an empty artifact list and does not itself ensure that all executed stochastic inputs were preserved.

## Technical and theoretical assessment

The dynamics and estimand are stated clearly, but the paper does not provide a theorem, identifiability result, detection lower bound, or policy-performance analysis. A theorem is not mandatory for an empirical benchmark paper, but then the empirical study must carry the contribution. At present neither theory nor experiments establish significance.

The dimensionless coordinate \(\rho=\alpha\sqrt{\Delta t}/\sigma\) is reasonable as a per-step drift-to-Brownian ratio for the stated discretization. It is not a complete nondimensional description: horizon, sensing radius, capture radius, collector speed, particle density, wall geometry, and observation history also govern difficulty. The manuscript acknowledges policy and resource dependence, which should remain prominent.

The use of one-based first-contact time with \(H+1\) for no contact is internally consistent. However, this metric has large mass at the censoring point and can discard post-first-contact information. The authors should retain it as primary if preregistered, but report capture probability, restricted mean time to first capture, and complete paired outcome categories as supporting diagnostics.

## Missing baselines and controls

The following are required before a main-track empirical claim is credible:

1. one stationary collector and four stationary collectors;
2. one random collector and four independent random collectors;
3. systematic area coverage and density-greedy movement;
4. a faithful single-collector analogue or adaptation of the closest mobile-collector strategies;
5. frozen local-flow policy with particle-velocity removed, shuffled, and history-limited ablations;
6. four independent local policies versus the identical policy class with a strictly bounded shared summary;
7. centralized full-state and true-field privileged references;
8. shared-parameter recurrent IPPO and a standard MAPPO implementation under matched observation, capacity, action, training, and tuning budgets;
9. collector-count and swept-area-matched controls;
10. timestep refinement or continuous collision detection;
11. field-orientation, initialization, horizon, sensing-radius, and capture-radius sensitivity checks;
12. paired versus unpaired uncertainty and observed common-random-number covariance.

Growing geometry additionally needs an area/perimeter-matched non-growing control and a frozen rule for attached-node radius, motion, wall interaction, and within-step activation.

## Unsupported or premature claims

The manuscript is commendably conservative, so there are few direct false claims. The following phrases nevertheless require tightening:

- “benchmark” currently denotes a proposed environment core, not a validated benchmark with tasks, datasets, baselines, reference results, and evaluation tooling;
- “measurement instrument” is aspirational until estimator calibration and repeated-run validation exist;
- “causal local information contract” is supported at the API level, but not yet for adapters or trained-policy pipelines;
- “physical finite-difference particle velocity” is inaccurate at reflecting boundaries unless the reflected path convention is incorporated;
- “internally frozen before performance run” is premature while the signal grid, `local_flow_v1`, inference construction, seed budget, tie mechanism, and matched wrapper are still unfrozen.

The manuscript does not claim a scientific result, and I found no fabricated result in the supplied package.

## Ethics

I see no immediate human-subject, privacy, fairness, or environmental-data concern in the synthetic benchmark. The paper should still include a short limitations/impact statement if positioning the system for robotic search or surveillance, since downstream uses may be dual-use. This is not a present rejection reason.

## Clarity

The manuscript is concise, unusually transparent about incompleteness, and mostly easy to follow. The transition order, estimand, and evidence-status paragraph are especially useful. Clarity would improve with a system diagram, an observation tensor table, a nearest-work comparison table, and a precise statistical-procedure box. The current title and abstract sound closer to a completed benchmark paper than the actual design-stage evidence warrants.

## Strongest arguments

### Strongest rejection argument

The submission's advertised contribution is an empirical multi-collector detectability benchmark, yet it contains neither the policy whose boundary is being defined nor any benchmark experiment, dataset, estimator, boundary result, coordination mechanism, or multi-agent comparison. Several primary-validity components are explicitly blocking. The supplied engineering core cannot substitute for the missing scientific contribution, and the active question does not yet isolate an AAMAS-specific multi-agent effect.

### Strongest acceptance argument

The work has the foundations of a useful and unusually auditable benchmark: explicit paired counterfactuals, scenario-level sampling, staged interception/aggregation outcomes, a careful leakage boundary, strong initial correctness tests, transparent claim status, and a literature map that sharply limits novelty. If fully executed, the combination could provide a reusable instrument for testing whether complementary local evidence and bounded communication change behavioral weak-signal exploitation.

## Required repairs

1. Complete the causal execution path: event-keyed ties, matched adaptive-policy wrapper, frozen `local_flow_v1`, config-driven runner, trajectory writer, schema validation, checksums, and analysis script.
2. Resolve endpoint-only capture through a continuous-contact rule or demonstrate timestep convergence at the canonical scale.
3. Freeze the exact signal grid, simultaneous inference construction, pilot-to-seed-budget rule, stopping rule, and criterion for a stable crossing; validate coverage in simulation.
4. Establish task nontriviality with stationary, random, coverage, density, closest-work adaptations, and privileged references before learned policies.
5. Make the central AAMAS claim genuinely multi-agent by isolating independent parallel search from bounded evidence sharing under matched collector count, information, policy capacity, action budget, and tuning budget.
6. Run sufficient independent scenario seeds, publish raw paired artifacts and failures, report uncertainty and pairing diagnostics, and perform the listed mundane-explanation checks.
7. Expand related work in the manuscript and compare the contribution directly against mobile-collector, local-sensing foraging, noisy multi-robot search, sensing-limit, CRN, aggregation, and MARL benchmark precedents.
8. Update every claim only after its evidence is generated; retain negative and non-monotone outcomes.

## Scores

These are confidence-calibrated reviewer-simulation scores on a 1--5 dimension scale and a 1--10 overall recommendation scale.

| Category | Score | Rationale |
|---|---:|---|
| Originality | 3/5 | The exact controlled combination appears potentially distinctive, but most components have strong precedents and the novel coordination measurement is unexecuted. |
| Significance | 2/5 | Potentially useful benchmark question, but no evidence yet that the task reveals a general or important multi-agent phenomenon. |
| Technical soundness | 2/5 | Several core choices are careful, but event-keyed ties, matched adaptive execution, contact discretization, and statistical calibration remain unresolved. |
| Empirical methodology | 1/5 | There is an evaluation plan but no scientific run, dataset, baseline result, uncertainty estimate, or analysis. |
| Theory | 2/5 | Formal dynamics and estimand are present; no theoretical result or analysis establishes a threshold or mechanism. |
| Clarity | 4/5 | Clear, compact, and honest, with some terminology and missing-detail issues. |
| Reproducibility | 3/5 for the engineering slice; 1/5 for the scientific study | The 44 supplied tests pass, but the proposed experiment cannot yet be reproduced because it has not been implemented or run. |
| Ethics | 4/5 | No immediate concern; a brief dual-use discussion would suffice. |
| AAMAS fit | 2/5 | Multi-agent setting is present, but multi-agent interaction or coordination is not the active demonstrated contribution. |
| Overall recommendation | **2/10 — Reject** | A promising but incomplete design-stage package, not a main-track empirical paper in its current state. |
| Reviewer confidence | **4/5** | High confidence about the absence of scientific evidence and current AAMAS-fit problem; moderate uncertainty about eventual novelty because the proposed study has not been executed. |

## Minimum changes that could alter the recommendation

The recommendation could move materially only with new evidence, not prose alone. At minimum, the authors must (i) complete and validate the paired execution/data/inference pipeline, including contact discretization; (ii) run a preregistered signal sweep with transparent baselines and uncertainty; and (iii) demonstrate an effect that is specifically multi-agent—most plausibly that a bounded shared summary shifts the stable signal-exploitation crossing relative to matched independent collectors, beyond collector count and swept area. A null coordination result could still support a useful benchmark paper if the task exposes a robust, nontrivial limitation and the negative result survives strong controls, but the paper would need to frame and evidence that contribution directly.
