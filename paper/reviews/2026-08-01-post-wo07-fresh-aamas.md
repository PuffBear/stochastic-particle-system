# Fresh AAMAS Review — Post-SPS-WO-07

**Date:** 2026-08-01  
**Reviewed package:** current compiled `paper/manuscript/main.pdf` only  
**Previous reviews, author strategy, ledgers, and desired scores supplied:** no

## Summary

The paper studies a simulated particle-capture task with four mobile collectors.
Its intended multi-agent question is whether replacing independent local
velocity estimates with a shared team-mean estimate improves fixed-horizon
unique-particle yield. The paper documents a sequence of diagnostic gates: a
first-contact endpoint is found to be saturated; a redesigned 67-step yield
endpoint shows action-contingent headroom for a privileged full-state
controller; a coupled-noise timestep diagnostic passes for the
oracle-versus-stationary contrast; and the final shared-versus-independent
diagnostic fails its preregistered joint gate. On eight seeds, the shared
controller's mean advantage is 1.75 particles, with only 4/8 positive
differences and a descriptive bootstrap interval of `[-0.375,3.75]`. The
authors consequently make no positive coordination, communication, learning,
or MARL claim.

The paper is commendably candid and contains thoughtful simulator-engineering
practices. However, in its current form it does not make a sufficiently
original, significant, or empirically established multi-agent contribution for
the AAMAS main track. It reads primarily as a detailed negative internal
mechanism audit.

## Strengths

- Unusually transparent treatment of negative results, failed gates,
  superseded endpoints, excluded runs, and evidentiary limits.
- Appropriate common-random-number matching across stochastic conditions.
- Stationary and true-field controls expose passive-transport and targeting
  explanations.
- Exact piecewise-specular contact handling, stable event-keyed tie-breaking,
  immutable records, and provenance hashes are sensible engineering decisions.
- Coupled Brownian refinement is substantially better than comparing unrelated
  trajectories across timesteps.
- The manuscript does not convert the failed diagnostic into a claim that all
  sharing is ineffective.
- The PDF is visually clean and legible.

## Major concerns

### 1. No demonstrated AAMAS-level multi-agent contribution

The central sharing diagnostic fails. No learning is performed, teammate
positions are unused, and the principal comparison is between scripted
controllers differing only in independent versus averaged velocity estimates.
There is no demonstrated strategic interaction, credit assignment, learned
communication, allocation protocol, or coordination phenomenon. A benchmark
could itself be a contribution, but that would require broader validation,
standardized tasks and metrics, an accessible implementation, and evidence that
it exposes meaningful multi-agent challenges.

### 2. Central intervention underspecified

The paper does not give a complete mathematical or algorithmic definition of
the bounded shared summary, independent comparator, or full-state oracle. It
needs the exact three values and bounds, per-collector construction and
normalization, missing-history rules, self-inclusion, communication timing,
action map, target assignment, receding horizon, and tie resolution. “Same
action rule and three numeric input slots” does not by itself establish capacity
matching.

### 3. Final experiment is small and inconclusive

Eight seeds, an interval spanning zero and effects above the two-particle
threshold, and only four positive differences support the procedural statement
that the frozen gate failed, but not a decisive scientific null. Calling the
mechanism closed is defensible as project management, not as proof of no effect.
The two-particle and 5/8 thresholds are not scientifically justified in the
paper, and no power calculation was performed for this final contrast.

### 4. Mundane explanations remain

The spatially uniform field makes global averaging an ordinary pooled-denoising
operation. A gain need not be coordination; a null may reflect estimator
construction. The full-state controller's advantage can come from target
positions and assignment. Movement can improve coverage without signal. The
current controls do not separate pooled denoising, synchronized motion, target
deconfliction, redundant pursuit, or true coordination.

### 5. Numerical validation misses the main contrast

The timestep study evaluates oracle minus stationary at one field strength, not
shared minus independent. It does not establish numerical robustness of the
paper's central contrast or across the wider parameter space. “Timestep
robustness for the redesigned endpoint” is therefore broader than the evidence.

### 6. Exactness and artifacts are not externally auditable

The paper gives neither a proof nor sufficient pseudocode and adversarial test
results for the repeated “exact” contact characterization. It reports tests,
hashes, immutable packages, and preregistration, but provides no anonymized
repository, artifact link, timestamped protocol, or supplementary specification.

### 7. Project history displaces archival content

Work-order identifiers, superseded analyses, procedural authorization, and a
process-control race occupy space needed for precise algorithms, absolute
outcomes, figures, robustness analysis, and a concise scientific narrative.

## Missing baselines and controls

The final endpoint needs random-motion, coverage, density-greedy, stationary,
and true-field baselines on the same seeds; shuffled, delayed, stale,
randomized, zero-message, own-estimate-duplication, and leave-one-out controls;
an equal-effective-sample-size estimator; one-versus-four collector scaling;
absolute signal/null yields; redundancy and allocation diagnostics; robustness
across task parameters; heterogeneous fields; timestep refinement of shared
minus independent; and, if MARL is claimed, strong IPPO/MAPPO and
centralized/decentralized baselines.

## Unsupported or overstated statements

- Timestep robustness is shown only for oracle minus stationary at one setting.
- “Exact” contact handling is asserted through tests rather than demonstrated
  formally or through accessible artifacts.
- “Preregistered” is not externally verifiable from the paper.
- “Capacity matched” lacks controller and computational-budget detail.
- “Full-state oracle” may be misleading for a bounded receding-horizon heuristic.
- Common-random-number paired evaluation is good practice but not itself a
  major causal-methodological contribution.
- Failing the progression threshold does not prove the effect is below two;
  the descriptive interval includes larger effects.

## Scores

Using 1–5 for dimensions:

- Originality: **2/5**
- Significance: **1/5**
- Technical soundness: **2/5**
- Empirical methodology: **2/5**
- Theory: **1/5**
- Clarity/presentation: **3/5**
- Reproducibility: **1/5**
- AAMAS relevance: **1/5**
- Reviewer confidence: **4/5**

**Overall: 2/10 — Strong Reject.**

## Strongest rejection argument

The paper finds no supported multi-agent effect, evaluates no learned or
strategic multi-agent method, and supplies neither a validated benchmark
artifact nor enough policy detail and empirical power to make the negative
result decisive. It therefore lacks both demonstrated AAMAS relevance and a
substantive archival contribution.

## Strongest acceptance argument

The work models unusually careful scientific behavior: it identifies endpoint
saturation, corrects a confounded estimand, uses matched stochastic streams,
tests contact and timestep issues, preserves failed diagnostics, and refuses to
overclaim. With a complete public benchmark and a methodology/reproducibility
framing, this could become useful to researchers building stochastic
multi-agent environments.

## Minimum changes that could alter the recommendation

1. Fully specify and release the simulator, policies, oracle, configs,
   preregistrations, and per-seed data.
2. Run a prospectively powered fresh-seed confirmation with uncertainty tight
   enough to resolve zero and the practical threshold.
3. Separate pooled denoising from communication using shuffled, delayed,
   randomized, and equal-effective-sample controls.
4. Establish timestep robustness for shared minus independent.
5. Test heterogeneous fields and multiple team sizes or regimes where
   information is genuinely distributed.
6. Reorganize around the resulting scientific contribution and move process
   history to supplementary material.

For an accept-level AAMAS paper, the reviewer would additionally expect a
clearly isolated and replicated coordination mechanism, a substantive
multi-agent algorithmic contribution, or a broadly useful and thoroughly
validated benchmark.
