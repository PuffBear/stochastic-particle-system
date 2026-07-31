# Detecting and Exploiting Weak Structure in Stochastic Particle Systems

## Research question

What is the weakest latent-field signal strength at which a team of locally observing collectors achieves a reliably positive matched improvement in pre-contact first-interception performance over otherwise identical no-signal episodes?

## Central hypothesis

For at least one transparent local-flow or team-flow policy, the paired lower confidence bound on the signal-minus-null change in first-interception performance crosses zero at a finite signal strength below the oracle boundary and above the random/coverage boundary.

## Estimand

The primary estimand is the matched signal-minus-null change in a protocol-frozen pre-contact first-interception statistic, using identical initial states, Brownian noise, field orientation/centre, and tie-breaking randomness. The exact statistic and threshold estimator must be frozen before the main sweep.

## Controlled factors and secondary analyses

- null, uniform, and vortex fields;
- fixed and growing capture geometry;
- independent and shared-summary information;
- post-contact cascade growth and false cascades;
- collector count and one scale axis.

These are not additional research questions.

## Mandatory baselines

Random, coverage, density-greedy, local-flow, team-flow, and oracle-field scripted policies precede shared-parameter recurrent IPPO and one standard MAPPO implementation.

## Validity gates

Deterministic seeding; exact matched counterfactuals; Brownian scaling by sqrt(dt); correct reflecting boundaries; permanent single-owner capture; growing aggregates represented as attached capture discs; no field leakage; pre-contact/first-contact/post-contact event separation; trajectory and manifest validation.

## Kill criteria

Narrow or stop the detectability-boundary claim if the oracle cannot exploit planted fields, null and signal pairs are not exact counterfactual matches, boundary estimates are dominated by arbitrary metric thresholds, or scripted local-flow policies cannot distinguish signal from null over a defensible signal range.

## Current stage

Repository and research-control layer initialized. Next: simulator skeleton and fail-closed correctness tests. No experiment result exists yet.
