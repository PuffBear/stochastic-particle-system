# SPS-WO-10 Research Lead handoff

**Date:** 2026-08-01

**Input snapshot:** `d4a497ab50021d2ed17289d1ba56cf420075947a`

**Decision:** authorize deterministic aggregation-runner integration only.
SPS-C04 remains proposed and unsupported; zero scientific episodes, no
correlation grid, and no endpoint decision are authorized.

## Smallest unresolved question

Can the repository execute self-only and all-to-all aggregation under the same
episode-frozen nonzero Gaussian field while proving that aggregation mode is
the sole structural difference and logging every nonlinear departure from the
analytic estimator model?

This is a software/provenance gate, not the paper's scientific experiment.

## Required causal contract

- Use a new aggregation-pair runner. Do not repurpose the legacy null-versus-
  signal runner or its schemas.
- YAML contains only serializable field parameters. Each arm derives the same
  integer field seed from its dedicated field stream and samples a separate but
  fingerprint-identical immutable field object.
- Both arms have identical initialization, complete Brownian tensor, field
  realization, event-key definition, encoder, decoder, action bounds, and
  observation contract. `aggregation_mode` is the only structural treatment.
- Sent messages are required to match only before state divergence or in
  forced-identical-state tests. Later differences are valid causal descendants
  of earlier actions and must not be mislabeled as a pairing failure.
- Teammate positions are disabled. Stable particle IDs, source positions,
  latent field velocities, and all diagnostic-only values remain outside
  `LocalObservation` and never enter the controller.

## Message and theory mapping

The frozen message fraction is `q_i=valid_count_i/nearest_particles_k`, not the
fraction of present slots. Log the raw mean, component clipping, sent and
received messages, own-empty status, received-message fallback, cross-agent
rescue, and zero-direction cancellation. A global message with positive `q`
may bypass an independently empty receiver's density fallback; that is a
treatment consequence, not a pairing defect.

For decision step `t>=1`, use the stable particle IDs selected at state `t` but
their retained positions immediately before transition `t-1 -> t`. Then

`B_ij = signal_strength^2 * mean C(X_p-X_q)`

and

`Omega_ij = (diffusion_sigma^2/dt) * |S_i intersection S_j|/(n_i n_j)`.

Compute the existing general risk difference `D` only if every set is nonempty
and particle reflection, message clipping, and fallback do not invalidate the
additive-noise interpretation. Step zero and every ineligible row must contain
an explicit reason rather than coerced numeric theory values. Collector
reflection is logged separately because it affects action-to-displacement
interpretation.

## Mandatory microcases

The bounded suite must cover field fingerprint equality; forced same-mode
identity; constant-message identity; opposing-message cancellation; all-empty
fallback; partial-empty cross-agent rescue; clipping; reflected-particle
ineligibility; collector reflection; exact shared-particle `Omega`; moving
geometry with fixed `d0`/`eta`; permutation equivariance; and immutable artifact
collision rejection. These are deterministic fixtures, not scientific seeds.

## Open scientific blockers

1. The literature veto is not closed.
2. The paper says `T=1.34`, while canonical `horizon*dt` is `400*0.02=8.0`.
   WO-10 must not resolve this silently; a later outcome-blind preregistration
   must prospectively freeze one physical endpoint and timestep-matching rule.
3. A prospective incidence/error-budget rule for analytic-ineligible rows is
   still required.

## Go/no-go rule

**GO** means only that a separate outcome-blind preregistration work order may
be designed. It requires all deterministic and legacy tests to pass, exact
pairing hashes, same-mode identity apart from labels, exhaustive nonlinear
classification, schema-valid immutable artifacts, and zero scientific output.

**NO-GO** follows from hidden information or capacity differences, field
resampling, unclassified clipping/reflection/fallback, schema mismatch,
nonreproducibility, or any legacy semantic regression. No-go triggers repair,
not a seed run.
