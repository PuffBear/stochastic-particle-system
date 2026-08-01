# Fresh AAMAS Main-Track Review — 2026-07-31 (Pre-Repair)

## Summary

This submission proposes a matched-counterfactual benchmark for asking when four locally observing mobile collectors can exploit a weak transport field in a stochastic particle system. Signal and no-signal episodes share initial conditions, Brownian forcing, nuisance variables, and policy randomness. The paper describes an exact within-step specular-contact solver, event-keyed ownership ties, artifact provenance, and a studentized simultaneous-bootstrap boundary rule. A 12-seed exploratory calibration finds no positive simultaneous lower bound over the tested signal grid. At the largest signal, stationary and random controls descriptively match or exceed the scripted local-flow policy. The paper explicitly acknowledges that its four collectors do not coordinate, that no learned policy is evaluated, that its exploratory seeds cannot support confirmation, and that it currently fails its AAMAS relevance gate.

The manuscript is admirably candid, but in its current form it is a diagnostic project report rather than an AAMAS main-track research contribution. It establishes neither the proposed detectability boundary nor a multi-agent mechanism, and its central estimand does not isolate exploitation by the policy from the effect of changing the environment itself.

## Strengths

1. The authors distinguish exploratory calibration from confirmation and avoid presenting negative lower bounds as a discovered boundary.
2. The matched signal/null construction and complete-seed-block resampling are sensible variance-control ideas.
3. The treatment of reflected within-step contacts and event-keyed ties addresses subtle reproducibility issues that are often ignored in particle simulators.
4. The manuscript is unusually explicit about adverse evidence: passive baselines match or exceed the focal policy, the privileged control does not separate, and the nominal team does not coordinate.
5. The paper clearly delimits several claims it does not make.

## Major concerns

### 1. The submission has no AAMAS-level multi-agent contribution

The four collectors run independent copies of the same scripted rule and ignore teammate positions. There is no cooperation, communication, strategic interaction, decentralized-learning problem, credit assignment problem, or learned policy. Replicating a single-agent controller four times does not create a multi-agent research question. The manuscript itself concedes this point. As a result, even a successful detectability-boundary estimate for the current treatment would primarily be a stochastic-control or sports/physics-style benchmarking result, not yet an AAMAS main-track result.

### 2. The primary estimand does not identify policy exploitation of latent structure

The estimand compares first-interception time under signal strength \(\alpha\) with first-interception time at zero signal while holding the policy fixed. A positive value can arise because the field passively carries particles into capture discs, irrespective of whether the policy senses or exploits the field. This is not a minor confound: stationary and random policies descriptively equal or outperform the focal local-flow rule. Therefore, the proposed boundary is a boundary for a combined environment-plus-policy effect, not a boundary for exploitation by locally observing collectors.

The essential contrast is missing. At minimum, the paper needs a policy-versus-passive treatment contrast within each signal level and then a matched difference-in-differences against the corresponding zero-signal contrast. A stronger design would directly randomize or ablate the information that carries the field signal while matching motion, action, and swept-area budgets.

### 3. The focal mechanism fails even as an exploratory control policy

At \(\rho=2\), the stationary, random, and purportedly privileged policies descriptively match or exceed `local_flow_v1`. This leaves several mundane explanations unresolved: passive flux dominates; motion reduces interception opportunities; the opposite-mean-velocity heuristic has the wrong sign; causal velocity estimates are too sparse or noisy; the sensor radius/horizon/action speed makes control ineffective; or first-contact performance is governed mainly by swept area and geometry. The privileged control's failure to separate is especially concerning because it suggests either a weak/incorrect oracle or a testbed in which the intended action advantage is absent. Scaling the same experiment to more seeds would not repair this mechanism failure.

### 4. The empirical evidence is exploratory, underpowered for the apparent effect scale, and not confirmatory

All scientific results use 12 calibration seeds and are explicitly ineligible for confirmation. No inferential comparison among policies is provided. The reported means are approximately 0.004--0.011, whereas the proposed target half-width is 0.05, several times larger than the observed effects. Thus the 24-seed reservation is not tied to a scientifically meaningful minimum effect and cannot resolve the question the paper asks. The very large studentized maximum critical value (6.187) at \(n=12\) also indicates unstable simultaneous inference. Synthetic false-crossing checks under two null generators do not establish operating characteristics under the actual zero-inflated/censored outcome distribution, nor do they establish power.

The design needs a simulation-based power analysis for a predeclared minimum meaningful policy-specific effect, followed by genuinely held-out seeds. It also needs uncertainty for every policy contrast, not descriptive rankings alone.

### 5. The system is insufficiently specified for reproduction

Several parameters or definitions required to reconstruct the experiment are absent or unclear, including the capture radius, the exact definition and normalization of \(b_z\), the mapping from the reported \(\rho\) grid to \(\alpha\), and enough policy/oracle pseudocode to reproduce all controls. The paper reports 79 passing tests and extensive manifests, but no supplementary artifact or released trajectory package is supplied to the reviewer. A statement that a provenance system exists is not itself reproducibility evidence. The claimed preregistration is also not linked to a timestamped public or submission-contained protocol.

### 6. Numerical validity remains open

The exact contact solver is exact only conditional on a single Euler step's piecewise-linear proposals. The manuscript correctly admits that this does not establish convergence of the stochastic dynamics. Because the response variable is first contact, it can be highly sensitive to time discretization. A coupled-noise timestep-convergence study is mandatory before interpreting even the exploratory policy ordering.

### 7. Novelty is not established

The related-work discussion contains only two references. That is inadequate to position the contribution against multi-robot search/coverage, pursuit and capture, decentralized partially observed control, collective foraging, common-random-number simulation optimization, sequential detectability, and benchmark methodology. The paper's strongest prospective contribution is an experimental instrument, but the manuscript neither surveys comparable instruments nor demonstrates that its combination of matching, exact contacts, and simultaneous boundary inference changes a substantive scientific conclusion.

### 8. The manuscript is not in a main-track submission form

The supplied paper uses a generic article layout and is only four pages including references. It reads as an internal status document: an entire section is titled "AAMAS relevance failure and blockers," future components are listed as later work, and the central conclusion is that no claimed result exists. This transparency is useful during research, but it is not a completed archival submission.

## Strongest rejection argument

The paper explicitly presents no confirmed scientific result and no multi-agent mechanism. Moreover, its primary signal-versus-null estimand conflates passive environmental transport with policy exploitation, a problem directly exposed by stationary and random controls matching or exceeding the focal policy. Consequently, the current experiment cannot answer its stated question in the intended sense, and the work does not fit the AAMAS main track.

## Strongest acceptance argument

The submission shows unusually strong scientific hygiene for an early benchmark: matched common-random-number episodes, explicit simultaneous inference, exact reflected contact handling, event-order-invariant tie breaking, provenance checks, and honest disclosure of adverse controls. If paired with a genuinely decentralized coordination mechanism and a correctly identifying contrast, this infrastructure could support a valuable and reproducible multi-agent benchmark.

## Missing baselines and controls

- A capacity- and action-matched stationary/passive comparator in an inferential policy-specific contrast, not only separate signal-versus-null descriptive means.
- A shuffled-velocity or delayed-velocity information ablation that preserves motion and observation volume while removing usable field direction.
- A full-state, action-feasible interception oracle that demonstrably outperforms passive baselines when control can matter.
- Capacity-matched independent versus coordinated policies using the same observation, action, compute, and communication budgets.
- Communication/no-communication and teammate-channel ablations.
- Standard decentralized multi-agent learning baselines if learning is part of the eventual claim (for example independent and centralized-training/decentralized-execution baselines).
- Coupled-noise timestep refinement and sensitivity to capture radius, horizon, sensor radius, collector speed, particle count, and field family.
- Multiple field structures; a spatially uniform drift may reduce the task to estimating a single global direction and does not demonstrate spatially distributed coordination.
- Outcome controls beyond first interception, such as a predeclared team capture curve or total captures, with care not to add endpoints post hoc.

## Unsupported or currently unverified claims

- Calling the system a reproducible boundary instrument is premature without a complete reviewer-accessible artifact and a successful end-to-end reproduction.
- The contact solver's exactness is argued at the within-step geometric level, but the submission contains no proof or supplementary validation available for review.
- The 79-test statement is not independently assessable from the supplied package.
- The novelty boundary is not credible with only two cited works.
- "Explicit multi-agent controls" overstates the current empirical design because no coordination treatment or inferential multi-policy contrast is evaluated.

## Mundane explanations that must be ruled out

- Passive transport, rather than local sensing, produces the signal/null improvement.
- Collector movement reduces stationary capture opportunities or moves collectors away from high-flux regions.
- Local velocity observations are too intermittent or noisy for the heuristic to estimate drift.
- The heuristic's sign or temporal alignment is wrong.
- First contact is dominated by initial geometry, swept area, or censoring.
- The chosen speed, horizon, sensing radius, and uniform field leave too little action leverage.
- Euler-step artifacts alter near-contact ordering despite exact chord-level contact checks.

## Clarity and presentation

The prose is clear and the negative-result boundaries are responsibly stated. The two tables are readable, and the PDF has no obvious clipping or overlap. However, the generic format, sparse related work, lack of a system diagram, absence of algorithm pseudocode, and status-report framing make it unsuitable as a finished main-track manuscript. The phrase "pre-contact first-interception performance" is also awkward because first interception is itself the contact event.

## Reproducibility and ethics

The described provenance practices are promising, but the supplied package is insufficient for independent reproduction. Exact configurations, executable code, environment lockfiles, raw per-seed outcomes, analysis scripts, and the preregistration record should accompany the paper. I see no material ethical concern in the stated simulation study; the main ethical obligation is accurate characterization of exploratory versus confirmatory evidence, which the current draft handles well.

## Confidence-calibrated scores

Scale for criteria: 1 = poor, 3 = adequate, 5 = excellent. Overall score: 1 = strong reject, 10 = strong accept. Confidence: 1 = low, 5 = high.

| Category | Score | Rationale |
|---|---:|---|
| Originality | 2/5 | Some potentially useful integration of matching, geometric contact handling, and boundary inference, but novelty is not established. |
| Significance | 1/5 | No supported boundary, policy advantage, coordination effect, or learning result. |
| Technical soundness | 2/5 | Careful definitions in places, but the estimand does not identify policy exploitation and numerical convergence is unresolved. |
| Empirical methodology | 1/5 | Twelve exploratory seeds, no held-out confirmation, no inferential policy contrasts, and an effect-insensitive seed target. |
| Theory | 1/5 | No theorem, proof, identifiability argument, or substantive theoretical result is supplied. |
| Clarity | 4/5 | Clear and unusually candid, though under-specified for reproduction and framed as a status report. |
| Reproducibility | 2/5 | Strong claimed provenance design, but key parameters and reviewer-accessible artifacts are missing. |
| Ethics | 5/5 | No evident ethical issue; exploratory limits are disclosed responsibly. |
| AAMAS fit | 1/5 | No multi-agent mechanism or evaluation is present. |
| Overall | **2/10 (Reject)** | The submission is not yet a completed AAMAS research contribution. |
| Reviewer confidence | **5/5** | The central deficiencies are stated directly in the manuscript and do not depend on subtle domain judgments. |

## Minimum changes that could alter the recommendation

1. Replace the current estimand with a preregistered contrast that isolates policy use of information from passive field effects, and demonstrate on held-out seeds that an action-feasible oracle separates from passive controls.
2. Introduce one precise, genuinely multi-agent mechanism--for example bounded sharing of local velocity summaries--and compare it inferentially with an identical-capacity independent policy under matched observation, action, compute, and communication budgets.
3. Run coupled-noise timestep convergence and repair any sensitivity before scientific inference.
4. Define a minimum meaningful effect, perform simulation-based power analysis for the actual paired outcome, and run a powered held-out experiment with multiplicity-controlled uncertainty for the key coordinated-versus-independent contrast.
5. Supply the complete simulator and analysis artifact, all missing environment and policy definitions, raw per-seed outcomes, and an executable reproduction path.
6. Expand the literature review enough to establish novelty and AAMAS fit, then rewrite the manuscript around one supported multi-agent claim rather than the present list of blockers.

Even these changes would not guarantee acceptance; they are the minimum needed for the work to become reviewable as an AAMAS main-track submission rather than an exploratory benchmark report.
