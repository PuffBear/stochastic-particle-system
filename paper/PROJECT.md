# When Shared Estimates Hurt: A Correlation-Scale Phase Boundary in Multi-Agent Collection

## Active research question

For four decentralized collectors that each broadcast one three-scalar local
velocity summary per control step, at what ratio

`eta = field correlation length / nominal collector spacing`

does all-to-all arithmetic averaging change the sign of its effect on distinct
team captures by physical time `T=1.34`, relative to capacity-matched
independent estimation?

This is the only active paper question. It asks for one grid-censored sign-change
boundary in one primary outcome. Estimator error, action alignment, and pursuit
overlap are mechanism diagnostics, not additional questions. Learned
communication, adaptive topology, target-intention messages, growing geometry,
and MARL remain inactive unless this scripted mechanism survives.

The complete design is in `paper/redesign_SPS_C04.md`.

## Why the paper changed

SPS-WO-07 validly tested the previous uniform-field three-number team mean. Its
shared-minus-independent effects were `[3,6,4,-3,0,0,5,-1]`, with mean `1.75`
and `4/8` positive seeds. That failed both frozen continuation thresholds. The
old claim SPS-C03 is dropped, not weakened or reinterpreted.

The mixed signs may motivate a new theory but do not establish hidden regimes.
Seeds 4001--4008 remain diagnostic-only and are permanently ineligible for the
new claim. The redesign uses a fresh field family, new work order, and future
fresh seed block.

## Frozen conceptual formalism

Let arena area be `A`, collector count be `M=4`, nominal spacing be
`d0=sqrt(A/M)`, and field correlation length be `ell_c`. The treatment axis is

`eta = ell_c / d0`.

Each agent forms the same message

`z_i = (local mean vx, local mean vy, valid-velocity fraction)`.

- Independent agents act from their own `z_i`.
- Globally sharing agents all act from the arithmetic mean of the four `z_i`.

The controller, three message slots, fallback, observation history, action
limits, initial state, Brownian tensor, field realization, tie randomness, and
physical horizon are matched. The only causal intervention is aggregation.

For fresh matched seed `s`, define

`Delta_s(eta) = Y_s(global, eta) - Y_s(independent, eta)`,

where `Y` is fixed-window unique team captures. The primary estimand is the
grid-censored zero crossing of `E[Delta_s(eta)]`. A positive paper requires
simultaneous evidence of a harmful low-correlation region and a beneficial
high-correlation region in the predicted order. Otherwise the primary result is
“no supported crossover on the frozen grid.”

## Theory target

For a stationary field and independent homoskedastic local-estimation errors,
global averaging reduces noise but introduces spatial-mismatch error. If
`S=sum_jk c_jk` is the sum of normalized cross-agent field correlations, the
agent-average one-component estimator-risk difference is

`R_global - R_independent = sigma_v^2 (1 - S/M^2) - tau^2 (1 - 1/M)`.

The theory work must derive the corresponding estimation crossover and connect
it to the non-additive unique-capture objective. Shared actions can increase
duplicated pursuit, so the yield crossover need not equal the estimator
crossover. This estimation-gain/action-diversity gap is the intended multi-agent
mechanism.

## Current implementation boundary

The repository already provides fixed-geometry capture, four collectors, causal
local velocity histories, the three-scalar summary, unique-yield outcomes,
event-keyed ties, and matched Brownian streams.

It does **not** yet provide:

- a field with a formally defined correlation length;
- receiver-specific graph communication;
- explicit message/link accounting;
- message-level estimation and action-diversity logs; or
- a matched topology runner.

The existing `vortex_scale` is an envelope width and may not be relabelled as a
correlation length. No scientific performance seed is authorized until the new
field, covariance recovery, communication intervention, diagnostics, and matched
stream microcases pass.

## Mandatory controls

1. capacity-matched independent estimation and all-to-all averaging;
2. stationary, pregenerated random, coverage, and density-greedy policies;
3. privileged true-local-field control and full-state assignment oracle;
4. shuffled messages and own-estimate duplication;
5. one collector and four independent collectors;
6. exact matching of message dimension, cadence, arithmetic, observations,
   actions, and stochastic streams;
7. coupled timestep validation at low, boundary, and high `eta` conditions;
8. learned baselines only after the scripted phase mechanism passes.

## Immediate dependency order

1. finish the primary-literature veto and formal derivation;
2. implement and validate an episode-frozen field with declared `ell_c`;
3. implement the pure three-scalar aggregation channel and diagnostics;
4. pass deterministic constant-field, opposing-region, denoising,
   permutation, bandwidth, and matched-stream tests;
5. freeze the `eta` grid, effect threshold, inference, seed cap, and stopping
   rule;
6. execute one fresh CPU diagnostic;
7. kill or preregister a separate independent confirmation.

No HPC request is justified at the current stage.

## Kill criteria

Stop the AAMAS redesign if the analytic crossover is absent in defensible
regimes; the field does not recover its declared covariance; deterministic
mechanism tests fail; communication changes anything besides aggregation; the
same correlation-scale/unique-yield crossover is already established; or a
fresh bounded diagnostic does not support one ordered negative-to-positive
crossing. Do not respond by adding seeds, learning a larger network, changing
the grid, or reusing WO-07 outcomes.
