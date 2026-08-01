# Detecting and Exploiting Weak Structure in Stochastic Particle Systems

## Research question

At `alpha=0.06`, with four collectors and a fixed physical window equal to 67
steps at `dt=0.02`, does one bounded three-number team velocity summary increase
unique team capture yield relative to an identical-shape controller whose three
message slots contain only the agent's own local estimate?

This is the only active research question. The earlier first-interception
boundary question was killed after the action-feasible oracle failed its frozen
gate. Aggregation, learned communication, field families, and scale remain
inactive directions unless a later Program Director decision establishes a
genuinely separable question.

## Frozen primary condition

The diagnostic condition compares `shared_summary` with
`capacity_matched_independent`. Both arms use four identical collectors, the
same action rule and three additional numeric input slots. The independent arm
fills those slots from the focal agent's own local estimate; the shared arm
receives the bounded team mean velocity and validity fraction. The field is
spatially uniform and the arena, initialization, noise and action limits are
matched:

`M=4, N=256, alpha=0.06, dt=0.02, evaluation_steps=67, sigma=0.06`.

Collector dynamics, sensing radius, action bounds, and initialization must be frozen in the environment contract before any performance sweep.

## Primary estimand

For matched seed `s`, let `Y_s(shared)` and `Y_s(independent)` be the numbers of
distinct particles captured by the four-collector team through the inclusive
fixed physical endpoint. The diagnostic paired contrast is

`Delta_s = Y_s(shared) - Y_s(independent)`.

The diagnostic gate is descriptive and cannot support the paper claim. If the
gate passes, a separately frozen confirmation uses a one-sided paired 95% lower
confidence bound for `E[Delta_s]`; a positive claim requires that bound to be
strictly above zero and the prespecified minimum relevant effect to be met.

## Matched counterfactual contract

Signal and null episodes must share the same initial state, pre-generated Brownian-noise tensor, field nuisance variables, and dedicated tie-breaking randomness. The only allowed causal difference is the planted field strength. Stateful random-number consumption after trajectory divergence is prohibited for the primary comparison.

## Central hypothesis

At the frozen `alpha=0.06` condition, the bounded team summary produces a
positive matched fixed-window yield contrast relative to the capacity-matched
independent controller. This is a mechanism-specific value-of-information
claim, not a claim that communication or MARL is generally beneficial.

## Closest prior work and novelty boundary

- Wang et al. (2025), *Mobile-collector capture of particles in a chaotic flow*, studies one mobile collector using local particle information in prescribed flows. We cannot claim the first mobile particle collector, first locally guided capture, or first vortex-flow collection benchmark.
- Löffler et al. (2023), *Collective foraging of active particles trained by reinforcement learning*, studies locally perceiving active particles trained with PPO. We cannot claim the first local-sensing RL particle-foraging task.

The defensible target is narrower: a reproducible matched intervention testing
whether one fixed, bounded team statistic has actionable value beyond an
identical-shape independent controller in a stochastic multi-collector task.

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

The deterministic simulator, exact fixed-geometry contact, causal observations,
strict matched runner, immutable artifacts, bounded independent/shared scripted
controllers, and the step-67 unique-yield endpoint are implemented. SPS-WO-05
established diagnostic action-contingent headroom: the full-state oracle exceeded
stationary by 9.375 captures on average with 8/8 positive diagnostic seeds.

SPS-WO-06 then passed its preregistered coupled-noise timestep gate. The
oracle-minus-stationary mean was 8.625 particles at both `dt=0.02` and
`dt=0.01`, with 0/8 seed-level sign changes; the mandatory informational
`dt=0.005` mean was 8.375. All correctness, coupling, provenance, and artifact
gates passed. This authorizes SPS-WO-07 only and is not coordination evidence.

SPS-C03 remains blocked. The shared-versus-independent SPS-WO-07 diagnostic has
not run. No learned baseline, empirical power result, power-sized confirmation,
or AAMAS coordination result exists. MARL implementations are downstream
engineering scaffolds and must remain inactive until the scripted mechanism
gate passes.
