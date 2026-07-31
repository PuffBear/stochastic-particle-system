# Detecting and Exploiting Weak Structure in Stochastic Particle Systems

## Research question

What is the weakest latent-field signal strength at which a team of locally observing collectors achieves a reliably positive matched improvement in pre-contact first-interception performance over otherwise identical no-signal episodes?

This is the only active research question. Aggregation, communication, field family, and scale remain controlled factors or secondary analyses unless a later Program Director decision establishes a genuinely separable paper.

## Frozen primary condition

The first estimable condition is `local_flow_v1`: four identical collectors receive only their own local sensor summary plus teammate positions; the field is spatially uniform; the arena and initialization protocol are fixed; and

`M=4, N=256, H=400, dt=0.02, sigma=0.06`.

Collector dynamics, sensing radius, action bounds, and initialization must be frozen in the environment contract before any performance sweep.

## Primary estimand

Let `T*` be the one-based first-interception step in `{1,...,H}`, with `T*=H+1` when no particle is intercepted. For matched seed `s` and signal strength `alpha`,

`D_s(alpha) = (T*_{s,null} - T*_{s,signal(alpha)}) / H`.

The primary effect is `delta(alpha)=E_s[D_s(alpha)]`. Positive values mean the signal episode intercepts earlier. The dimensionless signal coordinate is

`rho = alpha * sqrt(dt) / sigma`.

The reported boundary is grid-censored: the smallest preregistered tested `rho` whose one-sided simultaneous 95% lower confidence bound for `delta(rho)` is strictly above zero. If no tested point crosses, report right-censoring; if the weakest tested point crosses, report left-censoring. No interpolated primary boundary is permitted.

## Matched counterfactual contract

Signal and null episodes must share the same initial state, pre-generated Brownian-noise tensor, field nuisance variables, and dedicated tie-breaking randomness. The only allowed causal difference is the planted field strength. Stateful random-number consumption after trajectory divergence is prohibited for the primary comparison.

## Central hypothesis

For `local_flow_v1`, the paired lower confidence bound crosses zero at a finite preregistered signal level. This claim is policy-relative and grid-relative; it is not an information-theoretic detection limit.

## Closest prior work and novelty boundary

- Wang et al. (2025), *Mobile-collector capture of particles in a chaotic flow*, studies one mobile collector using local particle information in prescribed flows. We cannot claim the first mobile particle collector, first locally guided capture, or first vortex-flow collection benchmark.
- Löffler et al. (2023), *Reinforcement learning and optimal control of a minimal active particle foraging*, studies local-sensing reinforcement learning for active-particle foraging. We cannot claim the first local-sensing RL particle-foraging task.

The defensible target is narrower: a reproducible multi-collector instrument for estimating a policy-relative weak-field detectability boundary under exact signal/null counterfactual pairing, with explicit coordination and geometry controls.

## Mandatory baselines

Before learned policies:

1. stationary collectors / passive flux;
2. random motion;
3. area-coverage motion;
4. density-greedy motion;
5. local-flow scripted policy;
6. published-strategy adaptations from the closest mobile-collector work when implementable;
7. centralized full-state oracle.

For the AAMAS coordination claim:

1. one collector;
2. `M` independent collectors with no shared summary;
3. the same policy class with and without a bounded shared summary;
4. centralized full-state control;
5. shared-parameter IPPO;
6. one standard MAPPO implementation.

Growing-capture analysis additionally requires an area/perimeter-matched non-growing control.

## Validity gates

- deterministic named seed streams;
- pre-generated or event-keyed stochastic disturbances;
- exact matched signal/null initialization and Brownian noise;
- Brownian scaling by `sqrt(dt)`;
- correct reflecting boundaries, including arbitrary overshoot;
- permanent single-owner capture with dedicated tie-breaking randomness;
- attached-node growth active only from the next step;
- no field, future-noise, or global-state leakage into local observations;
- one-based first-contact semantics and explicit no-contact censoring;
- stationary and zero-signal limiting cases;
- trajectory, manifest, and analysis validation;
- fixed confirmatory signal grid and simultaneous inference rule.

## Kill criteria

Narrow or stop the active claim if:

- the oracle cannot exploit the planted field under the frozen task;
- exact null/signal counterfactual pairing fails;
- stationary or passive-flux controls explain the apparent improvement;
- the boundary changes materially under defensible metric encodings;
- `local_flow_v1` never crosses below the oracle reference across the frozen grid;
- a claimed team advantage vanishes when collector count, swept area, information, and policy capacity are matched;
- the task remains a replicated single-agent collection problem with no isolated multi-agent mechanism.

## Compute policy

Correctness tests and scripted pilots run on Codex cloud. No HPC request is justified until the simulator, observation contract, scripted baselines, seed audit, and pilot variance estimate pass. The first confirmatory budget must be derived from measured pilot variance and documented before execution.

## Current stage

The deterministic primitives, exact piecewise-specular fixed-geometry contact, event-keyed ties, capture-free reset, causal observations, strict matched runner, immutable schema-validated artifacts, frozen scripted controls, calibrated simultaneous estimator, 79-test suite, exploratory seed-level summaries, evidence-constrained LaTeX manuscript, and immutable fresh review now exist.

SPS-P02 is a 12-seed exploratory calibration only. It has no positive simultaneous lower-bound crossing through `rho=2`; stationary and random controls descriptively match or exceed `local_flow_v1`; and the four-collector rule is exactly four independent replicas. SPS-C01 therefore remains proposed, SPS-E01 is blocked, and no coordination, MARL, or AAMAS claim is available. Growing geometry, coupled-noise timestep convergence, a meaningful full-state interception oracle, a policy-specific passive-flux contrast, and a genuinely multi-agent bounded-sharing mechanism remain unresolved.
