# SPS-WO-11 Research Lead handoff

**Date:** 2026-08-01

**Input snapshot:** `f62b073cf2a5b1e283aff2e27ccd6ec7fae55c3d`

**Result:** endpoint and nonlinear-row design gates passed prospectively; zero scientific episodes.

## Physical endpoint

Freeze the SPS-C04 paper endpoint at `T=1.34`. State `s_0` is time zero;
decision step `j=0,...,66` produces swept transition `k=j+1`. Capture on
transition 67 is included and transition 68 is excluded. The primary mapping
is `(dt,horizon)=(0.02,67)`. Coupled numerical checks use
`(0.02,67),(0.01,134),(0.005,268)` with one finest Brownian path aggregated
upward and the same frozen field realization.

The canonical `400*0.02=8.0` setting remains a generic simulator default. It is
not the SPS-C04 endpoint. `T=1.34` was selected before SPS-C04 outcomes, passed
the WO-05 action-headroom and WO-06 coupled-timestep diagnostics, and avoids an
unvalidated longer window that may saturate unique capture.

## Outcome and mechanism contract

Primary unique yield retains every transition and matched episode. Clipping,
particle or collector reflection, empty summaries, fallback, rescue, and
cancellation never delete an outcome. Only a correctness or provenance failure
invalidates a matched seed--eta pair.

`B`, `Omega`, and scalar-component `D` are defined only when decision step is at
least one, all four collectors have a nonempty valid set, no selected valid
particle reflected on the preceding transition, no velocity component clipped,
and no receiver used fallback. Under the implemented independent identical
field components, the preregistered two-dimensional prediction is `D_2D=2D`.

Empty-summary rescue remains analytically ineligible but is a real treatment
consequence. Zero-direction cancellation is eligible if the other analytic
conditions hold. Collector reflection does not invalidate estimator covariance;
it excludes only the affected current collector-transition from realized-
displacement mediation. Outcome particle reflection does not invalidate the
current source-time estimator row or exact capture outcome.

## Frozen incidence budget

Compute every quantity separately for each arm and eta. Analytic eligibility is
divided by emitted post-history rows. Clipping uses `2M` message components;
own-empty and fallback use `M` agent-rows; reflected-valid incidence uses valid
selected-particle slots; collector reflection uses `M` action transitions;
rescue uses own-empty agent-rows; cancellation uses nonfallback agent-rows.
Action cosine and pursuit overlap use only pairs with defined actions or two
valid implied targets.

The gate requires analytic eligibility at least 80% in every arm--eta cell,
between-arm eligibility difference at most 10 percentage points, and no
individual seed-arm below 50%. Clipped components may be at most 1%; own-empty,
fallback, reflected-valid slots, and collector reflection may each be at most
5%. The collector-reflection cap applies only to realized-displacement
mediation. Rescue and cancellation have no cap.

A failed coverage threshold preserves the all-row yield estimand but kills the
covariance-supported mechanism and therefore blocks the current AAMAS claim
from confirmation. Do not discard rows, add seeds, or fit a post-hoc nonlinear
theory.

## Next action

After deterministic tests pass, design a separate preregistration work order
for the eta grid, effect threshold, simultaneous inference, fresh seed block,
stopping rule, and CPU budget. This handoff authorizes none of those choices.
