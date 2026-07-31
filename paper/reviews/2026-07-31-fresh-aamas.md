# Immutable Fresh AAMAS Review

## Summary

The submission proposes a benchmark and trajectory dataset for studying whether a small team of locally observing mobile collectors can exploit weak latent structure in the motion of many stochastic, non-learning particles. Signal strength is varied between matched signal-present and signal-absent episodes. The proposed environment includes uniform and curved latent fields, fixed and irreversible growing capture geometry, independent and shared-summary information conditions, transparent scripted policies, standard MARL baselines, and a privileged oracle. The intended primary claim is the existence of a finite policy-relative detectability boundary in pre-contact first-interception performance.

The central idea is potentially useful: common-random-number signal/null pairs, explicit separation of pre-contact detection from post-contact cascade amplification, and scripted/oracle controls could produce a clean benchmark. However, the submitted package is presently a research proposal rather than an AAMAS paper. It contains no implemented formalism, literature review, algorithms, dataset, experimental results, statistical analysis, figures, or empirical evidence for any proposed claim. Consequently, originality cannot be established and technical soundness cannot be evaluated beyond the design intentions.

## Strengths

1. **Promising experimental decomposition.** Separating evidence gathering, first interception, and post-contact cascade growth is scientifically well motivated. It could distinguish genuine signal exploitation from a lucky collision amplified by irreversible capture.

2. **Matched counterfactual design.** Holding initial states, stochastic increments, field parameters, and tie-breaking randomness fixed across signal/null episodes should substantially reduce variance. This is stronger than comparing independently sampled signal and null episodes.

3. **Appropriate transparent controls are planned.** Random, coverage, density-greedy, local-flow, team-flow, and oracle-field policies could distinguish task impossibility from MARL optimization failure.

4. **Good falsification instincts.** The project documents kill criteria, simulator invariants, leakage checks, and explicit null and reversed interpretations. The stated intention not to treat learned-policy failure as proof of signal impossibility is particularly important.

5. **Potentially useful artifact.** If released with validated trajectories, generator code, fixed splits, manifests, compact and full versions, the benchmark could support decentralized control, system identification, offline MARL, and event prediction.

## Major weaknesses

### 1. This is not yet a reviewable technical paper

The package states that no experiment has run and that even the primary statistic and threshold estimator remain unfrozen. There is no simulator implementation or validation evidence, no dataset, no baseline result, no figure, no uncertainty estimate, no manuscript narrative, and no bibliography. Every scientific claim is marked “proposed.”

This alone prevents acceptance in the main technical track. An experimental benchmark paper must demonstrate that the benchmark works, exposes a meaningful capability gap, supports reproducible measurement, and differs materially from established environments.

### 2. The primary quantity is not operationally defined

“Weakest latent-field signal strength” is not currently an invariant scientific quantity. Its numerical value will depend on:

- the normalization of \(b_z\);
- diffusion scale \(\sigma\), timestep \(\Delta t\), and horizon;
- particle density and collective-interaction strength;
- arena scale and boundary handling;
- collector speed, sensing radius, and total capture footprint;
- the chosen policy class;
- the definition of “first-interception performance”;
- the confidence level and crossing estimator.

The submission must define a dimensionless signal-to-noise parameter or otherwise prove that field scaling is standardized. It must also specify whether the outcome is time-to-first-contact, first-contact probability by a frozen horizon, a censored-event hazard, or another statistic. Without this, the claimed boundary can move under arbitrary unit or metric choices.

A discrete sweep cannot identify an exact “weakest” signal. At best, it brackets a crossing interval unless a justified response model is fitted.

### 3. AAMAS fit is presently fragile

The particles are fixed stochastic processes, and the primary question can potentially be answered by one collector or by four independent scripted flow followers. Merely having four controlled collectors does not establish a substantive multi-agent contribution.

The AAMAS case requires a result about decentralization, complementary local information, communication, interference, task allocation, or coordination. The current primary claim only asks whether a locally informed team beats its own matched null behavior. If a single local-flow heuristic produces the entire effect, this is primarily a stochastic-control or active-matter benchmark, not clearly an agents-and-multiagent-systems contribution.

A convincing submission should show that the multi-agent structure changes the detectability boundary under controlled information and resource budgets, rather than treating coordination as an optional secondary plot.

### 4. Originality is unestablished

There is no related-work section or citation ledger. The submission does not compare its environment with work on:

- decentralized multi-robot search and target interception;
- source seeking and plume/flow tracking;
- distributed detection and active sensing;
- particle-swarm and active-matter control;
- multi-agent tracking under partial observability;
- stochastic capture and aggregation processes;
- existing MARL particle-world benchmarks and trajectory datasets.

The proposed combination may be novel, but a combination of known ingredients is not itself evidence of originality. The paper needs to identify the nearest three to five benchmarks and show which measurable question none of them supports.

### 5. The statistical boundary estimator is underspecified and vulnerable

The proposed rule—where a paired lower confidence bound crosses zero—raises several unresolved issues:

- repeated testing across signal levels can produce a false first crossing;
- empirical performance need not be monotone in signal strength;
- adaptive seed allocation near an observed crossing can bias inference unless the procedure is frozen;
- censoring arises if no interception occurs by the horizon;
- particle observations within an episode are correlated, so particles cannot be treated as independent samples;
- orientations, field centers, and initial conditions create hierarchical variation beyond random seeds;
- a statistically positive but negligible difference may not be scientifically meaningful.

The protocol should freeze the experimental unit, minimum meaningful effect, sample-size calculation, monotonic or non-monotonic estimator, interpolation rule, simultaneous uncertainty procedure, treatment of censored episodes, and held-out evaluation protocol.

### 6. Several causal comparisons remain confounded

- **Number of collectors:** More collectors mechanically increase swept area and first-contact probability. Comparisons need matched total sensing footprint, capture footprint, movement budget, or an explicit normalization.
- **Information sharing:** A “team-flow” summary may reveal substantially more information than independent policies receive. Information bandwidth, delay, aggregation function, policy capacity, and training budget must be matched.
- **MAPPO:** A centralized critic during training does not itself provide communication during decentralized execution. MAPPO versus IPPO is not a clean test of shared information.
- **Growing geometry:** Any cascade advantage may follow solely from larger absorbing area or perimeter. A geometry-matched fixed or area-equivalent control is indispensable.
- **Field effect:** A latent field may simply push more particles through particular static locations. Stationary or field-agnostic coverage controls are needed to show active detection rather than passive flux exposure.
- **Training reward:** Training on total captured fraction emphasizes post-contact returns, while the primary evaluation is pre-contact interception. This mismatch could obscure or manufacture apparent detection differences.

### 7. The simulator formalism is incomplete

Equation (1) is schematic. The paper needs a complete stochastic game or Dec-POMDP specification: states, actions, observations, transition kernel, reward, communication channel, horizon, capture ownership, aggregate motion, collision rules, boundary conditions, and information available at every decision point.

The “attached capture discs” mechanic requires particular care. It is unclear whether captured particles remain fixed relative to a moving collector, whether chains can pass through particles or walls, how simultaneous contacts are assigned, and whether aggregate growth changes collector dynamics. These choices could dominate results.

Common Brownian increments across signal and null runs are a reasonable variance-reduction coupling, but after trajectories diverge they are not literal identical counterfactual trajectories. The paired estimand and coupling should be justified explicitly.

## Strongest rejection argument

The submission currently provides no completed scientific contribution: it is an unimplemented benchmark proposal with undefined primary metrics, no state-of-the-art positioning, and no evidence for any claim. Moreover, its primary question may be solvable without meaningful multi-agent interaction, making AAMAS relevance uncertain.

## Strongest acceptance argument

If implemented as described, the benchmark could offer an unusually controlled measurement instrument. The combination of exact signal/null random-number coupling, pre-contact versus post-contact decomposition, irreversible cascade diagnostics, transparent scripted and privileged references, and trajectory-level release could expose when distributed local information becomes behaviorally usable without conflating task impossibility with MARL training failure.

## Mandatory missing baselines

In addition to those already listed:

- a single-collector baseline;
- stationary/no-action and fixed-location interception baselines;
- a centralized full-state controller separate from the field oracle;
- independent collectors with matched total action and sensing budgets;
- communication-enabled and communication-free versions of the same policy;
- a centralized scripted team-flow estimator;
- a classical filter or local velocity-estimation controller;
- an area/perimeter-matched non-growing geometry control;
- a signal-present policy with shuffled temporal histories;
- a null condition retaining identical collective interactions;
- a passive-flux baseline measuring interception without adaptive movement.

## Required controls and diagnostics

- Dimensionless signal normalization and field-magnitude calibration.
- Held-out field orientations, centers, seeds, and initial configurations.
- Explicit protection against tuning the signal grid or policy on the test split.
- First-contact survival curves and censoring-aware analysis.
- Episode-level or hierarchical uncertainty, not particle-level pseudoreplication.
- Simultaneous uncertainty across the signal grid.
- Sensitivity to timestep, horizon, boundary rule, and one scale axis.
- Capture-area, perimeter, and mobility matching for the aggregation claim.
- Communication-bandwidth and policy-capacity matching.
- Tests showing that local observations cannot reconstruct privileged field parameters through accidental leakage.
- Simulator invariant tests with analytically or numerically checkable limiting cases.

## Unsupported or premature claims

All three ledger claims remain unsupported. In particular:

- Existence of a finite detectability boundary has not been demonstrated.
- “Beyond a fixed-radius volume effect” is not meaningful until the geometry-matched control is precisely specified.
- A shared-summary boundary shift cannot be attributed to coordination unless information, capacity, optimization budget, and centralized-training effects are separately controlled.
- The project cannot yet describe itself as introducing a dataset; only a dataset plan exists.
- The vision’s expectation that coordination helps most near the transition is a hypothesis, not an established mechanism.

## Mundane explanations that must be ruled out

1. The field increases passive particle flux through collector locations.
2. Four collectors simply sweep more area than one.
3. A scripted flow follower directly reads planted drift rather than solving a difficult detection problem.
4. Shared summaries leak global field direction.
5. Growing geometry improves capture solely through increased area or perimeter.
6. The estimated crossing is created by an arbitrary metric, confidence threshold, or signal grid.
7. Boundary reflections create detectable density artifacts.
8. Longer horizons make every nonzero drift detectable.
9. Learned baselines fail because of reward mismatch, insufficient history, or optimization choices rather than partial observability.
10. A result on one uniform or vortex parameterization does not generalize beyond that handcrafted field family.

## Theory

No theorem is necessary for a benchmark paper, but the current theoretical foundation is insufficient. A formal Dec-POMDP/stochastic-game definition and a dimensionless detectability parameter are required. It would also help to establish basic limiting cases—for example, zero-signal equivalence, privileged-policy monotonicity under standardized fields, and how passive interception scales with collector footprint—so that empirical effects can be interpreted against known references.

## Reproducibility

The planned reproducibility infrastructure is stronger than average: deterministic seeds, trajectory manifests, matched pairs, claims and experiment ledgers, fail-closed validation, and fixed splits are all good practices. However, reproducibility is currently only promised. The package contains no runnable simulator, environment lockfile, configurations, tests, raw trajectories, analysis scripts, or result hashes.

## Ethics

No material human-subject, privacy, or dual-use concern is apparent. The data are intended to be synthetic. The final artifact should document compute requirements and any environmental cost of the full dataset generation, but this is not a major concern at present.

## Clarity

The project documents are concise and mostly clear. The separation between the one primary question and controlled secondary analyses is helpful. Important terms—“reliably positive,” “weakest,” “first-interception performance,” “oracle boundary,” “team-flow,” and “geometry-matched”—remain undefined. The current package is not structured as a paper and has no literature-grounded argument.

## Scores

Using the latest publicly documented AAMAS-style categories:

- **Technical quality:** 2/5 — Weak. The design has merit, but no implementation, formal specification, or evidence is supplied.
- **Significance:** 2/5 — Unclear significance. A useful benchmark may emerge, but present impact and differentiation are unproven.
- **Presentation quality:** 3/5 — Acceptable. The proposal is readable, but it is not a complete manuscript and key quantities are undefined.
- **Originality:** 2/5 — Potentially interesting combination, but impossible to establish without a state-of-the-art comparison.
- **Reproducibility:** 2/5 — Strong plan, no runnable artifact or results.
- **AAMAS relevance:** 2/5 — Weak-to-moderate unless the central empirical contribution isolates a genuinely multi-agent coordination or distributed-information effect.
- **Overall rating:** 2/10 — Clear rejection in its current form.
- **Reviewer confidence:** 4/5 — High confidence about the incompleteness and methodological gaps; lower confidence about eventual originality because no literature review or results are provided.

## Overall recommendation

**Reject.**

This recommendation concerns the submitted state, not the potential of the project. The design could mature into a competitive benchmark paper, but a proposal and scientific ledger are not substitutes for a formalized, implemented, empirically validated contribution.

## Minimum changes that could alter the recommendation

1. Supply a complete manuscript with a precise Dec-POMDP/stochastic-system formalization and a literature-grounded novelty argument.
2. Freeze a dimensionless signal parameter, primary first-interception estimand, minimum meaningful effect, and statistically valid boundary estimator.
3. Release and validate the simulator with invariant tests and matched signal/null checks.
4. Complete the transparent baseline sweep, including single-collector, passive-flux, centralized, local-flow, and oracle controls.
5. Demonstrate a robust held-out boundary result with adequate episode-level uncertainty.
6. Show at least one clean multi-agent finding under matched resource, information, and capacity budgets; otherwise retarget the work away from AAMAS.
7. For any aggregation claim, include an area/perimeter-matched geometry control and separately report pre-contact, first-contact, and post-contact effects.
8. Provide runnable configurations, analysis code, raw or compact trajectories, dataset manifests, and exact result provenance.

A positive and robust result after these repairs could move the work into borderline territory; strong originality relative to neighboring benchmarks and a clean coordination result would be required for an accept recommendation.
