# SPS-C04 paper redesign: correlation-scale communication crossover

**Status:** conditional analytic and deterministic feasibility passed in
SPS-WO-09; no scientific seed run is authorized by this document. Standard
environment/config/runner integration remains incomplete.

**Predecessor:** SPS-C03 is permanently dropped after SPS-WO-07 failed its
frozen continuation gate. Seeds 4001--4008 remain diagnostic-only and cannot be
used to choose, confirm, or estimate the redesigned claim.

## Exact research question

> On a prospectively frozen grid of field-correlation-length to nominal-spacing
> ratios, for four collectors with the canonical square initialization and the
> same three-scalar local message encoder and action decoder, where—if
> anywhere—does all-to-all arithmetic averaging change from decreasing to
> increasing fixed-horizon unique team captures relative to self-only use of
> each collector's message?

This is one question. Field-estimation error, action alignment, and duplicated
pursuit are mechanism diagnostics; they are not additional research questions.
Reliability-weighted fusion, learned graphs, target-intention messages, growing
capture geometry, and MARL are not part of the primary question.

## Dimensionless treatment axis

Let the arena have area \(A\), let \(M=4\), and define nominal collector spacing

\[
d_0 = \sqrt{A/M}.
\]

For an episode-frozen stationary velocity field with declared spatial
correlation length \(\ell_c\), define

\[
\eta = \ell_c/d_0.
\]

The redesigned experiment varies only \(\eta\) on a frozen grid. This is a
benchmark-conditional threshold, not a geometry-invariant physical constant:
collector motion makes pairwise geometry endogenous after initialization.
Signal
amplitude, diffusion, sensing radius, action limit, particle density, collector
reset, horizon, message dimension, update frequency, and arithmetic budget are
matched across communication arms.

The existing `vortex_scale` parameter is **not** a correlation length and may
not be relabelled as one. A new field family with a defined ensemble covariance
must pass covariance-recovery tests before any performance pilot.

## Information and intervention contract

At time \(t\), collector \(i\) forms the same bounded local message used by the
closed diagnostic,

\[
z_{i,t}=(\widehat v^x_{i,t},\widehat v^y_{i,t},q_{i,t})
\in[-1,1]^2\times[0,1],
\]

where \(q_{i,t}\) is the valid local-velocity fraction.

- **Independent arm:** collector \(i\) acts from \(z_{i,t}\).
- **All-to-all arm:** every collector acts from the same arithmetic mean
  \(\bar z_t=M^{-1}\sum_i z_{i,t}\).

Both arms use the identical controller, density fallback, three input slots,
action bounds, observation histories, initial state, Brownian tensor, field
realization, and tie-breaking stream. The sole causal difference is message
aggregation. Teammate positions must be disabled in the primary observation or
held identical and proven behaviorally inert.

## Single primary estimand

For fresh matched scenario seed \(s\), communication arm \(a\in\{G,I\}\), and
correlation ratio \(\eta\), let \(Y_s^a(\eta)\) be the number of distinct
particles captured by the team through the inclusive frozen physical endpoint.
Define

\[
\Delta(\eta)=\mathbb E_s[Y_s^G(\eta)-Y_s^I(\eta)].
\]

On a preregistered grid \(\mathcal G\), the target is a grid-censored crossover

\[
\eta^*=\inf\{\eta\in\mathcal G:\Delta(\eta)>0\}.
\]

A crossover may be reported only if simultaneous uncertainty intervals support
at least one negative region and at least one positive region in the theory-
predicted order. Otherwise the correct result is “no supported crossover on the
frozen grid.” No interpolated threshold is a primary result.

## Theory spine

For one velocity component, let \(V=(V_1,\ldots,V_M)^T\) be the latent local
velocity summaries and \(E=(E_1,\ldots,E_M)^T\) their zero-mean errors, with
\(E\) independent of \(V\). Write

\[
B=\operatorname{Cov}(V),\qquad \Omega=\operatorname{Cov}(E).
\]

The average risk for self-only use is

\[
R_I=\frac{\operatorname{tr}(\Omega)}{M},
\]

whereas the average risk when every receiver uses the global arithmetic mean
is

\[
R_G=\frac{\operatorname{tr}(B)}{M}
-\frac{\mathbf 1^T B\mathbf 1}{M^2}
+\frac{\mathbf 1^T\Omega\mathbf 1}{M^2}.
\]

Therefore

\[
D=R_G-R_I=
\frac{\operatorname{tr}(B)}{M}
-\frac{\mathbf 1^T B\mathbf 1}{M^2}
+\frac{\mathbf 1^T\Omega\mathbf 1}{M^2}
-\frac{\operatorname{tr}(\Omega)}{M}.
\]

Negative \(D\) favors pooling for estimator risk; positive \(D\) favors
self-only use. Correlated errors can eliminate the usual denoising benefit, so
an independent-noise argument is not sufficient.

The benchmark does not observe a point sample. Conditional on particle
positions \(X_p\) and valid sets \(S_i\), the exact sensing-kernel covariance is

\[
B_{ij}=\frac{1}{n_i n_j}\sum_{p\in S_i}\sum_{q\in S_j}C_{\ell_c}(X_p-X_q).
\]

Away from reflection, clipping, and fallback, Brownian apparent-velocity error
with per-particle component variance \(\sigma_D^2/\Delta t\) gives

\[
\Omega_{ij}=\frac{\sigma_D^2}{\Delta t}
\frac{|S_i\cap S_j|}{n_i n_j}.
\]

SPS-WO-09 implements and deterministically tests these two conditional
matrices. Reflection, clipping, missing local summaries, and fallback are
nonlinear deviations that must be logged and bounded in the next work order.

The familiar point-sensor model is only a limiting illustration. Suppose

\[
\widehat v_i=v(x_i)+\epsilon_i,\qquad
\mathbb E[\epsilon_i]=0,\qquad
\operatorname{Var}(\epsilon_i)=\tau^2,
\]

with independent measurement errors, marginal field variance \(\sigma_v^2\),
and normalized spatial correlation \(c_{ij}=C(\|x_i-x_j\|)/\sigma_v^2\).
Let \(S=\sum_{j,k}c_{jk}\). Averaged uniformly over receiving agents, the
all-to-all minus local estimator risk is

\[
R_G-R_I
=\sigma_v^2\left(1-\frac{S}{M^2}\right)
-\tau^2\left(1-\frac1M\right).
\]

Thus pooling trades a noise reduction against spatial-mismatch bias. With a
squared-exponential field covariance,

\[
C(h)=\sigma_v^2\exp\!\left(-\frac{h^2}{2\ell_c^2}\right),
\]

the estimator-side sign is predicted jointly from \(\eta\), team geometry, and
the noise-to-field variance ratio. It cannot be predicted from \(\eta\) alone.
For four fixed square centers and equal independent errors, a unique point-
sensor crossover exists only when \(0<\tau^2/\sigma_v^2<1\). At
\(\tau^2/\sigma_v^2=0.5\), the ideal squared-exponential illustration gives
\(\eta^*\approx0.960\); changing the geometry or correlating the errors changes
that value. This calculation does **not** establish the team-yield claim; it
provides a conditional pre-outcome ordering prediction.

The team-utility bridge uses the non-additive unique-capture objective. In
general,

\[
\mathbb E[U]=\sum_p P\!\left(\bigcup_i\{i\text{ captures particle }p\}\right).
\]

The simpler \(2-P(\text{same target})\) identity applies only to a two-agent,
one-choice special case. Common messages can reduce team utility by increasing
action or pursuit overlap even when they improve an individual estimate. The
paper must quantify this estimator-gain versus diversity-loss decomposition
without treating either mediator as a second research question. A positive
high-correlation estimator result without a positive unique-yield consequence
does not support the paper.

## Falsifiable hypotheses

1. **Primary phase hypothesis:** \(\Delta(\eta)\) changes from negative to
   positive as \(\eta\) increases on the frozen grid.
2. **Ordering gate:** the supported signs follow the theory-predicted ordering;
   a non-monotone or reversed pattern kills the proposed one-boundary account.
3. **Mechanism gate:** low-\(\eta\) global averaging must increase local
   field-estimation mismatch and/or pursuit overlap, while high-\(\eta\)
   averaging must reduce estimation error. These are explanatory gates, not
   separate claims.

Exact numerical grid points, minimum relevant yield differences, simultaneous
inference, seed cap, and stopping rule remain unfrozen until deterministic
field, covariance, communication, and mechanism microcases pass.

## Mandatory baselines and controls

The primary contrast is only all-to-all versus the self-only aggregation
control with the same encoder, decoder, and three-scalar action input.
The following controls protect its interpretation:

1. stationary, pregenerated-random, and deterministic coverage policies;
2. privileged true-local-field control and centralized full-state assignment
   oracle, to separate estimator failure from action/task failure;
3. one collector and four independent collectors, to separate team size from
   communication;
4. episode-shuffled messages, to detect generic common-motion artifacts;
5. matched message dimension, broadcast cadence, arithmetic, observations,
   action limits, and random streams;
6. coupled \(\Delta t,\Delta t/2,\Delta t/4\) numerical validation at the
   eventual diagnostic conditions;
7. learned IPPO/MAPPO and learned-communication baselines only after the
   scripted phase mechanism survives. They cannot rescue a failed phase gate.

## Deterministic feasibility gates before seeds

1. **Passed in WO-09:** a stationary periodic Gaussian field has a declared
   \(\ell_c\), immutable episode realization, exact finite-basis covariance,
   invariant marginal variance, and reproducible fingerprint.
2. **Passed in WO-09:** a constant-field microcase makes independent and global aggregation
   directionally equivalent, aside from controlled noise reduction.
3. **Passed in WO-09:** an opposing-region microcase makes global pooling cancel incompatible local
   directions while independent estimates preserve them.
4. **Passed in WO-09:** a homogeneous noisy-message microcase shows the expected \(1/M\) noise
   variance reduction.
5. **Passed in WO-09:** the communication primitive is permutation equivariant
   and processes exactly three scalars per sender per step.
6. **Schema passed; runtime logging open:** message, adjacency, received-summary, estimator-error, action-similarity,
   pursuit-overlap, and unique-yield diagnostics are logged in a new schema;
   immutable trajectory schema v1 is not silently changed.
7. **Open:** a matched-topology runner must prove identical initialization,
   Brownian tensor, field realization, and tie stream across arms. The new field
   is callable but not yet wired through standard environment/config/runner
   reset paths.

WO-09 used zero scientific episodes. Its deterministic tests support mechanism
possibility and software correctness only; SPS-C04 remains unsupported.

## Kill criteria

Kill or redesign SPS-C04 without a scientific seed sweep if any of the
following occurs:

- the estimator theory has no plausible sign crossover under physically and
  statistically defensible parameters;
- covariance or deterministic opposing-region microcases fail;
- the purported correlation length is only a renamed vortex envelope;
- communication changes observation access, message count, action capacity, or
  random streams in addition to aggregation;
- the closest-work audit finds the same unique-capture crossover and causal
  decomposition already established.

After a fresh bounded diagnostic, kill the phase-boundary claim if:

- no negative-to-positive sign change is supported on the frozen grid;
- the ordering is reversed or materially non-monotone;
- passive flux, common motion, timestep sensitivity, or task infeasibility
  explains the contrast;
- the only positive result is lower estimator error without a multi-agent team
  utility consequence;
- the required precision exceeds the preregistered CPU seed cap.

Do not respond to a failed gate by learning a larger communication network,
adding seed blocks, changing \(\eta\), or reusing WO-07 seeds.

## Literature boundary

- Li and Guo (ICRA 2012) already compare all-to-all and limited communication
  for distributed source seeking: https://personal.stevens.edu/~yguo1/paper/ICRA12_ShuaiLi.pdf
- Atanasov, Le Ny, and Pappas already provide distributed algorithms for
  stochastic source seeking: https://arxiv.org/abs/1402.0051
- Elwin, Freeman, and Lynch already connect environmental correlation and
  communication radius in distributed monitoring:
  https://robotics.northwestern.edu/documents/publications/femrobots.pdf
- Nakamura, Santos, and Leonard already use Voronoi-neighbor communication and
  novelty-gated samples for decentralized learning of spatial fields:
  https://arxiv.org/abs/2208.01800
- Jiang and Lu already motivate selective attention because indiscriminate
  global sharing can impair cooperation:
  https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html
- Du et al. already learn correlated communication topologies at AAMAS:
  https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p456.pdf

Consequently, this paper cannot claim that local communication, topology
selection, correlation-aware estimation, or harmful communication is new. Its
conditional opening is the predicted sign crossover in a non-additive
multi-agent collection objective and its decomposition into estimation gain and
action-diversity loss.

## Dependency order

1. close the literature veto and derive the estimator/occupancy propositions;
2. specify and unit-test the correlation-length field;
3. implement the pure three-scalar aggregation intervention and diagnostics;
4. pass deterministic mechanism and matched-stream gates;
5. freeze the \(\eta\) grid, relevance threshold, inference, seeds, and cap;
6. run one fresh CPU diagnostic;
7. either kill the claim or preregister an independent confirmation;
8. only then consider manuscript expansion or learned baselines.

No HPC access is justified for steps 1--6.
