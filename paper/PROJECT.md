# When Shared Estimates Hurt: A Correlation-Scale Phase Boundary in Multi-Agent Collection

## Active research question

On a prospectively frozen grid, for four decentralized collectors with the
canonical square initialization and identical three-scalar message encoders and
action decoders, at what ratio—if anywhere—

`eta = field correlation length / nominal collector spacing`

does all-to-all arithmetic averaging change from decreasing to increasing
distinct team captures by physical time `T=1.34`, relative to self-only use of
each collector's generated message?

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

For latent local-summary covariance `B` and possibly correlated message-error
covariance `Omega`, the general agent-average one-component estimator-risk
difference is

`D = tr(B)/M - 1'B1/M^2 + 1'Omega1/M^2 - tr(Omega)/M`.

The earlier independent homoskedastic expression

`R_global - R_independent = sigma_v^2 (1 - S/M^2) - tau^2 (1 - 1/M)`.

is a special case. For the actual local particle averages, `B` is a double sum
of the declared field covariance over both sensors' valid particle sets, and
`Omega` includes their particle-set overlap. Shared actions can increase
duplicated pursuit, so the yield crossover need not equal the estimator
crossover. This estimation-gain/action-diversity gap is the intended multi-agent
mechanism. `eta` alone is not a universal predictor; geometry and error
correlation are explicit conditions.

## Current implementation boundary

The repository already provides fixed-geometry capture, four collectors, causal
local velocity histories, the three-scalar summary, unique-yield outcomes,
event-keyed ties, and matched Brownian streams.

SPS-WO-09 now provides:

- an immutable episode-frozen periodic Gaussian field with explicit finite-
  basis covariance, declared `ell_c`, invariant marginal variance, and a
  realization checksum;
- pure independent/all-to-all three-scalar aggregation with an identical
  encoder and decoder;
- exact conditional sensing-kernel and overlap-error covariance calculations;
- deterministic constant, opposing, denoising, permutation, and provenance
  tests; and
- a strict message-diagnostic schema.

SPS-WO-10 now additionally provides a strict deterministic self-only versus
all-to-all runner, matched stream/field/event-key hashes, same-mode identity,
and runtime message, covariance, geometry, nonlinear-event, action, and yield
diagnostics. It does **not** provide scientific performance evidence.

SPS-WO-11 conditionally passes the threat-oriented novelty veto at confidence
0.65 and freezes the paper endpoint at inclusive `T=1.34` (67 transitions at
`dt=0.02`). It also freezes an all-row yield rule and a prospective nonlinear
coverage budget for the covariance-supported mechanism. These are design and
validity decisions, not scientific evidence.

The existing `vortex_scale` remains ineligible. WO-09 and WO-10 ran zero
scientific episodes; all scientific claims remain unsupported.

## Mandatory controls

1. self-only aggregation and all-to-all averaging with the same encoder,
   decoder, and three-scalar action input;
2. stationary, pregenerated random, coverage, and density-greedy policies;
3. privileged true-local-field control and full-state assignment oracle;
4. shuffled messages and own-estimate duplication;
5. one collector and four independent collectors;
6. exact matching of message dimension, cadence, arithmetic, observations,
   actions, and stochastic streams;
7. coupled timestep validation at low, boundary, and high `eta` conditions;
8. learned baselines only after the scripted phase mechanism passes.

## Immediate dependency order

1. freeze the `eta` grid, minimum relevant effect, simultaneous inference,
   fresh seed block, seed cap, and stopping rule without reading SPS-C04 outcomes;
2. encode the independent confirmation firewall before any diagnostic launch;
3. only then execute one fresh bounded CPU diagnostic;
4. audit the frozen nonlinear-coverage and timestep gates before interpretation;
5. either kill the phase claim or preregister an independent confirmation;
6. consider learned baselines only after the scripted mechanism survives.

No HPC request is justified at the current stage.

## Kill criteria

Stop the AAMAS redesign if the analytic crossover is absent in defensible
regimes; the field does not recover its declared covariance; deterministic
mechanism tests fail; communication changes anything besides aggregation; the
same correlation-scale/unique-yield crossover is already established; or a
fresh bounded diagnostic does not support one ordered negative-to-positive
crossing. Do not respond by adding seeds, learning a larger network, changing
the grid, or reusing WO-07 outcomes.
