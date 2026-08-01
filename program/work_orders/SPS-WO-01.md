# SPS-WO-01: Primary Simulator and Metric Gate

## Scope

This work order freezes the smallest high-information simulator and metric gate for the stochastic-particle benchmark. It does not change the established research question:

> What is the weakest latent-field signal strength at which a team of locally observing collectors achieves a reliably positive matched improvement in pre-contact first-interception performance over otherwise identical no-signal episodes?

No performance experiment may be interpreted scientifically until every mandatory test in this work order passes.

## 1. Primary condition

To make “weakest signal strength” identifiable rather than policy- and environment-dependent, use:

- Policy: frozen deterministic `local_flow_v1`, run independently by all four collectors.
- Information: local observations and teammate positions only; no shared team summary.
- Field: uniform field; orientation sampled from the frozen scenario-seed distribution and copied into the null counterfactual.
- Geometry: fixed capture geometry.
- Canonical environment: \(M=4\), \(N=256\), \(H=400\), \(\Delta t=0.02\), `diffusion_sigma=0.06`, reflecting unit square, with the remaining parameters from the supplied canonical specification.
- Event: the first capture of any particle by any collector.

The canonical `diffusion_sigma` remains `0.06` from the supplied specification unless a later documented pilot revises it.

This condition does not alter the research question. It fixes the canonical test through which the question is answered. `team_flow`, shared summaries, vortex fields, growing geometry, and scaling remain secondary analyses.

## 2. Exact primary metric and estimand

For scenario seed \(s\), field strength \(\alpha\), and condition \(c\in\{0,\alpha\}\), define

\[
T_{s,c}^* =
\begin{cases}
\text{first capture step}\in\{1,\ldots,H\}, & \text{if a capture occurs},\\
H+1, & \text{otherwise}.
\end{cases}
\]

Define the normalized first-interception score

\[
S_{s,c}=\frac{H+1-T_{s,c}^*}{H}.
\]

Therefore:

- capture on step 1 gives \(S=1\);
- capture on step \(H\) gives \(S=1/H\);
- no capture gives \(S=0\).

The matched seed-level improvement is

\[
D_s(\alpha)=S_{s,\alpha}-S_{s,0}
=\frac{T_{s,0}^*-T_{s,\alpha}^*}{H}.
\]

The primary estimand is

\[
\delta(\alpha)=\mathbb{E}_{s}[D_s(\alpha)],
\]

where the expectation is over the frozen scenario distribution of initial states, Brownian forcing, and field orientations. This threshold-free statistic is bounded in \([-1,1]\) and means “the fraction of the episode horizon saved before the first interception.”

“Reliably positive” means that the one-sided 95% simultaneous lower confidence bound for \(\delta(\alpha)\), computed by resampling complete scenario-seed blocks jointly across all tested strengths, exceeds zero.

The reported boundary is grid-censored:

\[
\alpha^*_{\mathrm{grid}}
=\min\{\alpha_j:L_j>0\}.
\]

Do not interpolate a precise continuous boundary in the primary analysis. Report \((\alpha_{j-1},\alpha_j]\); left-censor if the first positive grid point already qualifies, and right-censor if none qualifies. Logistic or isotonic crossings may be reported only as sensitivity analyses.

## 3. Required assumptions

1. A signal/null pair shares initial particle and collector states, the complete Brownian tensor, orientation or centre draw, policy RNG, and keyed tie-breaking randomness.
2. The pair differs only in `field.signal_strength`.
3. Scenario seeds—not particles, time steps, collectors, or unmatched episodes—are the independent sampling units.
4. `local_flow_v1` is frozen before the main sweep and receives no true field parameters or global particle state.
5. No particle begins inside a capture region.
6. Capture order and first-capture semantics are frozen.
7. Fixed and growing geometry are identical through first contact; aggregation cannot influence the primary metric.
8. The chosen strength grid spans at least one nonreliable and one reliable region. Otherwise the boundary is censored rather than estimated.
9. A positive field effect on uninformed policies is possible because drift can physically move particles toward collectors. Random and coverage controls therefore govern the interpretation, although they are not part of the primary estimand.

## 4. Falsifier

The primary detectability-boundary claim fails if, over a pre-frozen defensible strength range:

- no strength gives \(L_j>0\) for `local_flow_v1`;
- positive points are isolated or nonpersistent enough that a boundary interpretation is indefensible;
- random or coverage obtains an equal or larger matched improvement, meaning passive transport explains the apparent effect;
- the oracle does not exploit the planted field at high-but-feasible strength, indicating failed calibration; or
- paired-counterfactual, no-leakage, or first-contact invariants fail.

The final two cases are protocol failures, not scientific null results.

## 5. Mandatory simulator and metric tests

1. The same seed and configuration reproduce states, actions, captures, and metrics exactly.
2. Matched pairs have identical initial states, Brownian tensors, orientation or centre draws, policy RNG provenance, and tie RNG provenance.
3. `null` and `uniform(signal_strength=0)` produce identical full trajectories.
4. Brownian displacement variance per coordinate is \(\texttt{diffusion_sigma}^2\Delta t\) within a seeded statistical tolerance.
5. Uniform-field displacement equals \(\alpha\Delta t(\cos\theta,\sin\theta)\) before reflection.
6. Reflection handles one wall, corners, and multi-wall overshoot exactly.
7. Hand-built microcases verify capture at step 1, capture at step \(H\), and no capture; their \(S\) and \(D\) values must match hand calculations.
8. No capture is possible at reset.
9. Independent observations contain no field label, strength, orientation, global state, or out-of-radius particles.
10. Under matched actions and randomness, fixed and growing runs are identical through and including first capture.
11. Capture is single-owner and permanent; ties are reproducible.
12. The analysis resamples matched scenario blocks and never treats individual paired episodes as independent.
13. A bounded smoke pilot verifies oracle feasibility and checks whether random or coverage receives passive drift benefits.
14. Failure of any test blocks SPS-E01 and permits only simulator repair.

## 6. Secondary analyses outside the primary question

- Aggregation: post-first-capture counts, false cascades, and geometry-matched controls.
- Coordination: `local_flow` versus `team_flow`, and independent versus shared-summary information.
- Generalization: vortex fields and held-out orientations.
- Scaling: collector count and at most one environmental axis.
- IPPO and MAPPO: benchmark participants after scripted-policy validation.

These analyses may explain or qualify the primary finding, but they must not be combined into a conjunctive research question.

## 7. Prohibited claims at this gate

Do not claim:

- that a detectability boundary exists;
- that collectors infer or classify the latent field, because the metric establishes behavioral exploitation only;
- that growing geometry improves first interception, because it cannot do so by design;
- that coordination helps;
- that vortex, scaling, or out-of-distribution generalization works;
- that MARL learns the task or outperforms scripted methods;
- that failure of `local_flow_v1` proves the signal is fundamentally undetectable;
- a continuous or physically universal value of \(\alpha^*\);
- hydrodynamic realism or strategic particle behavior;
- any scientific result from smoke tests; or
- external preregistration, because this is only an internally protocol-frozen gate.
