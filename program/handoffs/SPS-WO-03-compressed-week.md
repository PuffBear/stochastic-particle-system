# SPS-WO-03 Compressed-Week Handoff

**Date:** 2026-07-31  
**Status:** engineering and exploratory calibration complete; confirmatory evidence round blocked  
**Claim status:** SPS-C01 remains proposed; SPS-P01/P02 cannot support or reject it

## Delivered

- Exact piecewise-linear fixed-geometry contact over rectangular specular reflections, including arbitrary multi-wall overshoot.
- Earliest-contact ownership with event-keyed deterministic ties invariant to unrelated event consumption and array order.
- Strict matched null/signal runner that rejects hidden treatment differences and verifies initialization, Brownian tensor, field nuisance, policy randomness, and tie provenance.
- Immutable closed-schema trajectory, pair-summary, and manifest artifacts with SHA-256 checksums.
- Frozen stationary, pregenerated-random, coverage, density-greedy, `local_flow_v1`, and privileged oracle policies.
- Seed-blocked studentized max-bootstrap simultaneous lower bounds, first/persistent grid censoring, and deterministic calibration.
- SPS-P01 guarded-contact diagnostic and SPS-P02 exact-contact replication over 144 matched pairs each.
- Updated claims/experiment/decision/expansion ledgers and evidence-constrained manuscript.

## Verification

Python 3.12.13 and NumPy 2.3.5. `PYTHONPATH=src python3 -m unittest discover -s tests -v` passes 79 tests; compileall and the research-program structure checker pass. Inference calibration used 2,000 outer null simulations per model and 2,000 inner bootstrap draws. Wilson 95% familywise-error upper bounds were 0.04951 for correlated continuous null effects and 0.01046 for zero-inflated symmetric discrete null effects, both below the frozen 0.07 acceptance threshold.

## Exploratory calibration result

At rho `[0.10, 0.25, 0.50, 1.00, 2.00]`, `local_flow_v1` mean gains were `[0.005625, 0.003750, 0.007083, 0.008958, 0.011042]`. Simultaneous one-sided 95% lower bounds were all negative: `[-0.015330, -0.024523, -0.019081, -0.019440, -0.020788]`. The exploratory boundary is therefore right-censored beyond rho=2.0. This is not a claim test because seeds 1001--1012 were designated for calibration.

At rho=2, descriptive mean gain was 0.01104 for local flow, 0.01313 stationary, 0.01292 pregenerated random, 0.01292 privileged oracle, 0.00771 coverage, and 0.00625 density-greedy. No multiplicity-controlled policy contrast was preregistered, so these rankings are diagnostics only.

SPS-P02 replaced 28,929 guarded reflection-pair diagnostics with exact specular paths and changed 0 of 144 first-interception outcomes. This is a same-seed correctness regression, not independent confirmation.

## Hard no-go findings

1. `local_flow_v1` ignores teammate information, so the four-collector primary treatment is exactly four independent replicas and cannot establish coordination, communication, MARL, or AAMAS relevance.
2. Passive stationary and random controls descriptively match or exceed local flow at the strongest signal; a positive signal/null effect alone cannot be interpreted as learned or inferred flow exploitation.
3. Euler/Brownian within-step trajectory convergence remains unmeasured even though wall reflection contact is now exact for the numerical path.
4. Pilot seeds cannot be reused for a confirmatory claim. The mechanical precision rule suggested 24 independent seeds, but the fresh reviewer found the 0.05 target half-width is coarser than observed effects of roughly 0.004--0.011; this run would be structurally uninformative and is not authorized.

## Safe next work

- Freeze a genuinely multi-agent bounded evidence-fusion policy and an identical-budget independent ablation.
- Add coupled-noise timestep convergence at fixed physical horizon and field strength.
- Preregister a policy-contrast family separating passive flux, random swept area, density seeking, and local velocity use.
- Do not train IPPO/MAPPO or request HPC until the scripted mechanism clears these gates.
