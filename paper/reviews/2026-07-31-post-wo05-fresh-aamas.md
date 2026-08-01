# Fresh AAMAS Main-Track Review — 2026-07-31 (post-WO-05)

## Review scope

This review considers only the frozen submission package consisting of
`paper/manuscript/main.tex` and `paper/manuscript/main.pdf`. No previous review,
rebuttal, internal record, code, raw output, author intention, or desired score
was consulted.

## Summary

The submission develops a matched-counterfactual benchmark for mobile
collectors intercepting stochastic particles in a reflecting unit square. It
carefully pairs signal and null episodes, defines exact within-step contact for
piecewise-specular Euler paths, and records event-keyed ownership ties. The
manuscript reports that an independently applied local-flow rule did not beat
passive controls in a 12-seed exploratory first-contact study. It then diagnoses
first-contact saturation and changes the endpoint to unique team captures over
67 steps. On eight fresh diagnostic seeds, a privileged full-state oracle
outperforms stationary and true-field-only controls under that redesigned
endpoint. The actual proposed multi-agent treatment—replacing each collector's
independent local velocity estimate with a bounded team-mean estimate—has not
been executed. The paper explicitly claims no coordination, learning, MARL,
timestep-convergence, power, or confirmatory result.

## Strengths

1. **Unusually clear claim discipline.** The manuscript sharply distinguishes
   engineering checks, exploratory diagnostics, and confirmatory evidence. It
   reports adverse findings and repeatedly states what is not established.
2. **Thoughtful variance control and provenance design.** Matching initial
   states, Brownian forcing, nuisance variables, and policy randomness is a
   sensible way to estimate small within-seed effects. The described checksum,
   schema, and immutable-write discipline is stronger than is typical for an
   early benchmark paper.
3. **A useful diagnosis of endpoint saturation.** The contrast between a
   saturated first-contact endpoint and a fixed-horizon yield endpoint is
   scientifically plausible and clearly explained. The full-state oracle is a
   reasonable upstream test of whether legal actions can matter at all.
4. **Technically careful event semantics.** The distinction between exact
   within-step contact for the discretized dynamics and actual timestep
   convergence is correct and responsibly stated.
5. **Readable presentation.** The prose, notation, and tables are generally
   understandable, and the PDF has no visible clipping, overlap, or broken
   glyphs.

## Major weaknesses

### 1. The paper does not answer its own research question

The first section asks whether a shared bounded team-mean velocity estimate
improves unique captures relative to independent local estimates. No outcome
for either side of that comparison is reported. The only positive result is
that a privileged full-state oracle beats two non-oracle controls on eight
diagnostic seeds. That establishes action-contingent headroom, not the value of
information sharing, coordination, decentralized decision making, or learning.
As submitted, this is a carefully documented pre-experiment rather than a
completed research paper.

### 2. AAMAS fit is presently inadequate

Four replicated controllers do not by themselves constitute a multi-agent
contribution. The evaluated local rule ignores teammates, the proposed shared
rule is unexecuted, and there is no coordination mechanism, task allocation,
strategic interaction, communication-learning problem, or MARL baseline. Even
if the team-mean treatment later produces a gain, in the stated spatially
uniform field it may amount to pooled denoising of a common vector rather than
a distinct multi-agent mechanism. The paper must explain what specifically is
multi-agent about the information structure and demonstrate that the effect
cannot be reproduced by giving each independent agent an equal-quality local
estimator or a centralized non-coordinating denoiser.

### 3. The empirical evidence is diagnostic and too small for the intended claim

The positive fixed-horizon result uses eight seeds, privileged state, and
descriptive bootstrap intervals. There is no frozen minimum effect or powered
seed budget for the actual shared-versus-independent contrast, no independent
confirmation set, and no timestep-convergence run. The paper supplies no
empirical estimate of the proposed coordination effect. The 12-seed earlier
calibration concerns a different endpoint and policy question and cannot fill
this gap.

### 4. Endpoint redesign creates a serious adaptive-design threat

The endpoint was changed after the first-event oracle failed. The new 67-step
window has a physical interpretation, but this does not remove the risk that
the endpoint was selected because it produced a favorable oracle diagnostic.
The manuscript needs a clean separation between endpoint-development seeds and
all estimation/confirmation seeds, plus robustness over a preregistered range
of plausible horizons. Otherwise a result at exactly step 67 could reflect
window selection rather than a stable mechanism.

### 5. The benchmark mechanism is underdeveloped

The primary field is spatially uniform, collectors share symmetric action and
sensor budgets, and the shared statistic is a simple team mean. This makes the
intended phenomenon narrow and potentially trivial: pooling more noisy samples
should improve estimation of one common direction. The paper does not provide
a theory of when pooling should help, when it should hurt, how correlation and
communication limits enter, or why capture yield requires genuinely collective
behavior. There is no ablation separating estimator variance reduction from
spatial coordination or reduced search overlap.

### 6. Missing baselines and controls prevent attribution

At minimum, the primary experiment requires:

- stationary, pre-generated random, and coverage controls under the new
  fixed-horizon endpoint;
- independent-local versus shared-mean policies with strictly matched action,
  observation, compute, and policy-capacity budgets;
- a centralized pooled-estimation but non-coordinating control;
- an independent policy given the same effective sample count or estimator
  variance, to isolate communication from denoising;
- shuffled, delayed, corrupted, and bandwidth-matched message controls;
- no-teammate and teammate-position ablations;
- target-deconfliction or assignment baselines that can distinguish shared flow
  estimation from spatial coordination;
- standard decentralized multi-agent learning baselines, such as IPPO and
  MAPPO, if the paper wishes to make a MARL claim;
- a full-state upper bound and a true-field-only upper/control policy evaluated
  on the same primary seeds and endpoint;
- timestep, seed-count, horizon, sensing-radius, drift-strength, noise, and team-
  size robustness.

### 7. Reproducibility is asserted but not available to a reviewer

The manuscript describes 89 tests, manifests, digests, trajectories, and an
R1 evidence package, but the frozen package supplies none of these artifacts or
an anonymized repository/supplement. Important implementation details of the
full-state oracle, shared-summary rule, bootstrap, and capture solver are not
sufficiently specified for independent reproduction from the paper alone.
Claims about test coverage and artifact identity therefore cannot be checked.

### 8. Submission readiness and format are poor

The PDF contains conspicuous red `PENDING` instructions, explicitly labels the
central claim blocked, and ends with an unexecuted work plan. It uses a generic
`article` layout rather than an identifiable AAMAS proceedings template. The
bibliography is very small for a paper spanning stochastic hitting,
multi-robot search, distributed sensing, communication, and MARL. The final
reference page also has substantial unused space. These issues reinforce that
the package is not yet a submission-ready main-track paper.

## Strongest rejection argument

The submission's sole research question concerns the causal effect of sharing
a team statistic, but that treatment has never been run. The reported positive
result only shows that a privileged full-state oracle can improve a redesigned
capture endpoint. There is therefore no multi-agent result to evaluate and no
evidence supporting the paper's intended contribution to AAMAS.

## Strongest acceptance argument

The strongest case is as a benchmark/methodology paper: the authors show rare
discipline in matching stochastic counterfactuals, defining exact event
semantics, diagnosing passive-advection confounding, and using a privileged
oracle to falsify an unsuitable endpoint before proceeding. If accompanied by
a complete, public, reproducible benchmark and a decisive multi-agent
information-sharing study, this could become a useful empirical contribution.
The current package, however, stops before that contribution is demonstrated.

## Unsupported or insufficiently supported claims

- The phrase "all frozen ... gates passed" is not independently verifiable from
  the supplied package.
- The assertion that the first-event failure reflects endpoint saturation is a
  plausible interpretation, not uniquely established; initial geometry,
  capture radius, policy quality, and estimator mismatch are alternative
  explanations.
- The fixed-horizon result establishes headroom only for the particular
  privileged oracle and selected horizon; it does not yet establish that the
  proposed decentralized observation or communication structure can exploit
  that headroom.
- Calling the artifact a "multi-collector benchmark" is descriptively fair,
  but any implication of a multi-agent scientific contribution is premature.
- The literature-based novelty boundary is not convincing with only eight
  references and no close comparison table.

## Mundane explanations that must be ruled out

1. Shared averaging helps only because it increases effective sample size for
   estimating a spatially uniform vector.
2. Any yield difference is caused by all collectors moving in a common
   favorable direction, not coordination or communication.
3. The 67-step result is specific to the chosen traversal-time window.
4. Performance is driven by initial collector lattice placement and passive
   particle flux through capture discs.
5. The oracle gain comes from privileged target positions and assignment, not
   from the information proposed for the actual treatment.
6. Results depend materially on Euler step size or missed/altered stochastic
   hitting behavior.
7. Capture gains reflect duplicated pursuit, tie-breaking, or target ownership
   semantics rather than more useful team coverage.

## Technical and presentation questions

1. What exact bits are communicated by the shared-summary policy, how often,
   and with what latency? Is the team mean broadcast by a central entity or
   computed through a decentralized protocol?
2. Does every collector receive exactly the same mean and therefore take the
   same directional action? If so, what prevents the method from collapsing to
   replicated common control?
3. How are samples weighted when collectors observe different numbers of valid
   particle velocities? Is the statistic a mean of agent means or a mean of all
   particle samples?
4. Why is one sensing-radius traversal the scientifically relevant evaluation
   horizon, and is the conclusion stable before and after step 67?
5. What is the capture radius? It is central to the saturation diagnosis but is
   not clearly stated in the frozen-system section.
6. How does the full-state oracle assign targets and determine action
   feasibility under Brownian uncertainty? The verbal description is not
   enough to reproduce it.
7. Are captures removed from future dynamics, and do collectors interact or
   collide? These choices affect the multi-agent structure.
8. What is the precise primary field (b_z(x)), and why is a spatially uniform
   field an adequate test of distributed collective sensing?

## Confidence-calibrated scores

The following use a 1--10 scale where higher is better, except reviewer
confidence, which uses 1--5.

| Category | Score | Rationale |
|---|---:|---|
| Originality | 4/10 | The matched-counterfactual and event-semantics package is careful, but the proposed shared mean is elementary and its scientific effect is untested. |
| Significance | 3/10 | A rigorous stochastic multi-collector benchmark could matter, but the current evidence does not establish a result of broad interest. |
| Technical soundness | 4/10 | Definitions are mostly careful, but the central experiment, timestep validation, power analysis, and confirmation are absent. |
| Empirical methodology | 3/10 | Strong pairing and diagnostic gates are offset by eight exploratory seeds, adaptive endpoint development, missing baselines, and no primary-treatment run. |
| Theory | 2/10 | The manuscript gives notation and dimensionless ratios but no substantive theory for communication or coordination benefit. |
| Clarity | 6/10 | The paper is candid and readable, though dominated by project history and blockers rather than a completed contribution. |
| Reproducibility | 4/10 | The described provenance is promising, but no reviewer-accessible code, supplement, raw data, or executable artifact is supplied. |
| Ethics | 8/10 | No evident human-subject, privacy, or societal-risk issue; claims are appropriately restrained. |
| AAMAS fit | 2/10 | No evaluated multi-agent mechanism, cooperation result, or MARL result is present. |
| Overall | **2/10 — Reject** | The submission is scientifically responsible but incomplete; it does not answer its stated question. |
| Reviewer confidence | **5/5** | The decisive limitation is explicit throughout the manuscript and does not depend on a subtle interpretation. |

## Minimum changes that could alter the recommendation

1. Execute a separately preregistered, powered, seed-disjoint
   shared-versus-independent experiment on the redesigned endpoint only after
   passing coupled-noise timestep convergence.
2. Demonstrate a nontrivial and robust positive effect against the attribution
   controls above, especially equal-effective-sample denoising and centralized
   non-coordinating baselines.
3. Show that the effect persists across a preregistered horizon range, multiple
   team sizes, signal/noise regimes, and at least one spatially heterogeneous
   field where decentralized information distribution matters.
4. Provide a clear multi-agent formulation: agents, local information,
   communication protocol and budget, decentralized execution constraints,
   interaction mechanism, and hypotheses about when sharing helps or hurts.
5. Add competitive scripted and learning baselines appropriate to the claim;
   include IPPO/MAPPO only if a MARL contribution is actually pursued.
6. Release an anonymized reproducibility package containing code, configs,
   manifests, seeds, raw/processed outputs, and one-command reproduction for
   every table and figure.
7. Replace the progress-history narrative with a completed-result narrative,
   remove all `PENDING` text, use the required AAMAS format, and substantially
   expand the nearest-work comparison.

## Overall recommendation

**Reject (2/10), confidence 5/5.** The engineering discipline and negative-
result transparency are commendable, and the new endpoint passes a useful
oracle headroom diagnostic. Nevertheless, the only question framed as the
paper's contribution has not been tested, and the package contains no
demonstrated coordination, communication, or MARL effect. A complete,
reproducible shared-information study that isolates genuinely multi-agent value
from pooled denoising is the minimum development that could change this
recommendation.
