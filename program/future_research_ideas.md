# Future Research Ideas

Inactive directions only; none may consume active-paper resources without a Program Director decision. Each entry is unvalidated.

## SPS-FR-001 — Retired duplicate

- **Status:** Retired into SPS-FR-009 on 2026-07-31. FR-009 is the result-grounded, falsifiable version and includes the required matched-budget and dropout controls.

## SPS-FR-002 — Event-keyed counterfactual random numbers

- **Provenance:** SPS-WO-01 through SPS-WO-03; `src/particle_benchmark/dynamics/capture.py`; SPS-P02.
- **Motivating result:** Event-keyed ownership is implemented and unit-tested, but all 144 SPS-P02 pairs recorded zero tie decisions; the pilot did not empirically exercise it.
- **Question:** Under deliberately high-overlap conditions, does event-keyed ownership remain invariant to unrelated prior event consumption while a stateful resolver does not?
- **Falsifiable hypothesis:** Event-keyed tie draws indexed by scenario, step, particle, and eligible-collector set are invariant to unrelated event-consumption order while naïve stateful draws are not.
- **Possible contribution:** An order-invariant event-randomness stress protocol, only if a material effect appears beyond contrived unit cases.
- **Minimal experiment:** forced-tie and divergent-control-flow families with collector permutations and injected unrelated prior events.
- **Required compute:** CPU-only unit tests and a bounded Monte Carlo diagnostic.
- **Workplan:** measure ownership divergence and paired-estimator variance; retain all failures; compare against the already implemented keyed resolver.
- **Kill criterion:** no material bias or variance effect outside contrived microcases.
- **Candidate venue:** simulation/reproducibility venue; not AAMAS absent a multi-agent scientific consequence. Verify current venue details before planning.
- **Status / priority:** validation backlog, 5/10; no longer an implementation blocker.

## SPS-FR-003 — Aggregation beyond geometric area

- **Provenance:** supplied benchmark vision; SPS-WO-02 engineering limitation.
- **Motivating observation:** The primary environment now uses point particles and fixed capture radius. Attached-node radius, aggregate-wall behavior, and captured-particle display motion are not frozen.
- **Question:** Does attached growth produce post-contact cascade acceleration beyond an area/perimeter-matched static control?
- **Falsifiable hypothesis:** With accessible area and perimeter matched, attached growth still changes the post-contact cascade distribution.
- **Possible contribution:** Separation of topological memory from simple capture-area expansion.
- **Minimal experiment:** first freeze attached-node kinematics and wall semantics, then compare growing and non-growing geometry under identical pre-contact trajectories and matched accessible perimeter.
- **Required compute:** CPU pilot followed by a seed budget derived from pilot variance.
- **Workplan:** preregister geometry-matched controls; decompose first contact, cascade size, and false cascades; test scaling across density.
- **Candidate venue:** AAMAS only if aggregation interacts with team coordination; otherwise complex-systems or simulation venues, subject to current deadline verification.
- **Status / priority:** `unvalidated`; 6/10 and inactive until the primary fixed-geometry benchmark passes.

## SPS-FR-004 — Nonstationary and moving latent fields

- **Provenance:** supplied benchmark vision; deferred uniform-field limitation.
- **Question:** How rapidly can a local team adapt when field orientation, centre, or topology changes within an episode?
- **Falsifiable hypothesis:** bounded memory improves matched post-change interception without harming stationary-field performance.
- **Possible contribution:** Detectability-versus-adaptation phase diagram with bounded memory.
- **Novelty threat:** standard online filtering may explain all benefit.
- **Kill criterion:** no adaptation benefit after matched memory and tuning budgets.
- **Candidate venue:** robotics or stochastic control; verify current details before planning.
- **Workplan / status:** freeze change process and adaptation metric; inactive until the uniform task is valid.

## SPS-FR-005 — Strategic or learned particle evasion

- **Provenance:** passive-particle limitation in SPS-P02.
- **Question:** How does the boundary change when particles respond strategically to collectors?
- **Falsifiable hypothesis:** response-aware policies improve matched interception against fixed evasion classes.
- **Possible contribution:** Genuine pursuit-evasion game extending the passive-particle benchmark.
- **Novelty threat:** reduces to standard pursuit-evasion without a particle-system contribution.
- **Kill criterion:** no separable population or stochastic-structure mechanism.
- **Candidate venue:** AAMAS only for a genuine strategic multi-agent game; otherwise robotics.
- **Workplan / status:** freeze evader information/action budgets and equilibrium concept; inactive.

## SPS-FR-006 — Heterogeneous collector teams

- **Provenance:** homogeneous-team limitation and AAMAS relevance audit.
- **Question:** When do heterogeneous sensing and actuation capabilities outperform homogeneous allocation under the same total resource budget?
- **Falsifiable hypothesis:** a prespecified heterogeneous allocation lowers a matched boundary under equal total resources.
- **Possible contribution:** Role emergence and capability allocation near weak signals.
- **Novelty threat:** advantage may be simple hardware-budget reallocation.
- **Kill criterion:** effect vanishes under total sensing/action/parameter matching.
- **Candidate venue:** AAMAS if role interaction is isolated; otherwise robotics.
- **Workplan / status:** preregister resource accounting and matched ablations; inactive.

## SPS-FR-007 — Offline MARL from paired trajectories

- **Provenance:** immutable paired-runner artifact contract; no released training dataset yet.
- **Question:** Can paired signal/null trajectory datasets support reliable offline policy selection near a detectability boundary?
- **Falsifiable hypothesis:** a prespecified selector ranks held-out policies better than unpaired or behavior-only controls.
- **Possible contribution:** Counterfactual benchmark for offline multi-agent evaluation.
- **Novelty threat:** dataset may lack action coverage and the primary task is not yet multi-agent.
- **Kill criterion:** unsupported coverage or no held-out ranking improvement.
- **Candidate venue:** offline RL/benchmark venue; AAMAS only after multi-agent relevance.
- **Workplan / status:** define coverage diagnostics and held-out policy suite; inactive.

## SPS-FR-008 — Catchability-aware scale transfer

- **Provenance:** canonical-task generalization limitation; SPS-P02; SPS-WO-04; 2026-07-31 pre-repair fresh AAMAS review.
- **Motivating observation:** The old axis, `rho = alpha*sqrt(dt)/sigma`, measures per-step drift relative to Brownian motion but not whether a collector can physically catch an advected particle. At the old `rho=2` point, `kappa = alpha/v_max` is approximately 7.07, so target advection is much faster than collector motion. This can mix observability with catchability.
- **Research question:** At fixed sensing and geometry, do first-interception curves collapse across physical rescalings when indexed by the catchability ratio `kappa = alpha/v_max`?
- **Falsifiable hypothesis:** After holding the remaining preregistered dimensionless groups fixed, curves from at least three physical rescalings have a maximum absolute discrepancy no larger than 0.01 in normalized restricted mean first-interception time when plotted against `kappa`.
- **Possible contributions:** A catchability-aware benchmark axis; an explicit separation between sensing difficulty (`rho`) and control authority (`kappa`); out-of-distribution splits that do not silently change the physical task.
- **Nearest literature / novelty risk:** Hein et al. (2016), `https://arxiv.org/abs/1512.04217`, already studies physical sensing limits through nondimensional regimes. Standard dimensional analysis may fully explain the collapse; the novelty would need to be an empirically useful two-axis benchmark protocol, not the existence of dimensionless groups.
- **Minimal experiment / data / compute:** Three scale-equivalent CPU configurations at `kappa in {0.25, 0.5, 1.0}`, with coupled scenario seeds and stationary plus full-state-oracle controls. Reuse no confirmatory claims from SPS-P02.
- **Two-week validation plan:** Days 1–3 derive and unit-check all dimensionless groups; days 4–6 construct scale-equivalent configs; days 7–10 run an 8-seed diagnostic; days 11–14 test the frozen 0.01 collapse tolerance and audit confounding by horizon, capture radius, sensor radius, density, and wall crossing time.
- **Full workplan:** Extend only after the active oracle and timestep gates pass; preregister rescalings and tolerances; run held-out seeds; publish absolute and normalized event-time curves; reject any collapse obtained by changing censoring or endpoint definitions.
- **Kill criterion / dependency / risk:** Kill if the curves do not collapse, if collapse follows trivially from established nondimensionalization without a benchmark consequence, or if the active task is killed at the oracle-feasibility gate. Depends on SPS-WO-04 Gates 2–3. Main risk is an overparameterized similarity analysis with too few independent physical regimes.
- **Candidate conference and verified deadline:** MODELSWARD 2027 is a conditional fit for model-driven benchmark transfer; its official call lists a regular-paper deadline of **15 September 2026** (`https://modelsward.scitevents.org/CallForPapers.aspx?y=2027`, checked 2026-07-31). Do not target it without a supported transfer result.
- **Status / priority:** `unvalidated`, inactive, 6/10. This refines the existing scale-transfer entry; it is not a new paper authorization.

## SPS-FR-009 — Bounded evidence fusion as the actual multi-agent question

- **Provenance:** SPS-P01; SPS-P02; SPS-WO-04; 2026-07-31 pre-repair fresh AAMAS review; `paper/literature.md` focused repair.
- **Motivating observation:** `local_flow_v1` ignores teammate information, so four collectors are independent replicas. The negative SPS-P02 calibration cannot identify coordination. SPS-WO-04 therefore freezes one three-number message: clipped team-mean local velocity (two values) plus team validity fraction (one value).
- **Research question:** At fixed catchable dynamics, does one bounded shared team-velocity summary improve passive-adjusted first-interception time relative to the identical-shape independent controller?
- **Falsifiable hypothesis:** At at least one preregistered nonzero catchable point, the mean paired shared-minus-independent contrast is at least 0.01 normalized-horizon units, has a positive simultaneous 95% lower bound, persists at stronger catchable points, and survives message-shuffled and leave-one-agent-out ablations.
- **Possible contributions:** A narrowly causal value-of-information result; a capacity-, observation-, action-, and arithmetic-budget-matched test of decentralized evidence fusion; a negative-result protocol if bounded pooling dilutes local evidence.
- **Nearest literature / novelty risk:** CIMAX (`https://arxiv.org/abs/1903.05444`) already establishes collective information maximization with local communication; Foerster et al. (2016), `https://proceedings.neurips.cc/paper_files/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html`, and Wang et al. (2020), `https://proceedings.mlr.press/v119/wang20i.html`, establish learned and bandwidth-constrained communication. The fixed summary is a mechanism probe, not a novel communication algorithm.
- **Minimal experiment / data / compute:** Eight diagnostic seeds on `kappa in {0.25, 0.5, 1.0}` after oracle and timestep gates; identical three-slot inputs for shared and independent controllers; passive, shuffled-message, and leave-one-out controls; Codex CPU only.
- **Two-week validation plan:** Days 1–2 freeze leakage and capacity accounting; days 3–5 unit-test permutation invariance and agent-ID equivariance; days 6–8 run the 8-seed diagnostic; days 9–11 estimate block-resampled power at a 0.01 effect; days 12–14 either preregister held-out confirmation or record a coordination-null decision.
- **Full workplan:** Proceed only after the full-state oracle establishes action-contingent headroom and coupled-noise refinement is stable; simulate type-I error and power; select the smallest passing seed count in `{16,24,32,48,64}`; run new held-out seeds; report absolute outcomes, passive-adjusted effects, bandwidth, failures, and all ablations.
- **Kill criterion / dependency / risk:** Kill the AAMAS mechanism if oracle feasibility fails, passive/shuffled controls explain the effect, the shared contrast is below 0.01 or non-positive, timestep ordering reverses, information leaks, or 64 seeds provide less than 80% simulated power. Depends on SPS-WO-04 Gates 1–3 and 5. Principal risk: global averaging may erase useful local heterogeneity or synchronize harmful actions.
- **Candidate conference and verified deadline:** AAMAS 2027 is the intended fit only for a surviving genuinely multi-agent result; the official call lists abstract **1 October 2026** and paper **8 October 2026** deadlines (`https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/`, checked 2026-07-31).
- **Status / priority:** `unvalidated`, 10/10. This is a mirror of the active mechanism gate, not authorization for a second paper or MARL training.

## SPS-FR-010 — Passive-adjusted policy value and oracle feasibility

- **Provenance:** SPS-P02 `baseline_summary.json` and `primary_analysis.json`; SPS-WO-04; 2026-07-31 pre-repair fresh AAMAS review.
- **Motivating exploratory result:** At `rho=2` on 12 calibration seeds, mean signal/null gain was 0.01104 for local flow, 0.01313 for stationary, and 0.01292 for both random and the true-field upstream control. These non-confirmatory descriptions show that signal/null gain alone can be passive transport and that “move upstream” is not an action-feasible interception oracle.
- **Research question:** In a catchable regime, does `local_flow_v1` reduce horizon-censored first-interception time beyond the reduction achieved by stationary collectors?
- **Falsifiable hypothesis:** The passive-adjusted effect `A_s(local_flow,alpha) = G_s(local_flow,alpha) - G_s(stationary,alpha)` is positive and exceeds velocity-slot shuffles at a persistent catchable point, after a centralized full-state action-feasible oracle first passes its feasibility gate.
- **Possible contributions:** An identifying passive-adjusted estimand for stochastic transport tasks; an ordered oracle-feasibility test that separates malformed tasks from weak local estimators; a principled negative result if control has no headroom.
- **Nearest literature / novelty risk:** Wang et al. (2021), `https://doi.org/10.1103/PhysRevE.104.064203`, establishes stationary capture units; Wang et al. (2025), `https://pmc.ncbi.nlm.nih.gov/articles/PMC12331103/`, explicitly compares mobile and stationary collection and decomposes carrier advection from controlled relative motion. The passive-adjusted contrast may be good experimental hygiene rather than a standalone novelty.
- **Minimal experiment / data / compute:** Diagnostic seeds 1001–1008, `kappa in {0.25,0.5,1.0}`, absolute restricted mean first-contact time plus `A_s`; stationary, random, coverage, density, local flow, velocity shuffles, action reversal, true-field-only control, and full-state receding-horizon oracle; CPU only.
- **Two-week validation plan:** Days 1–3 instrument valid tracks, action alignment, path length, walls, censoring, and swept area; days 4–6 pass deterministic oracle microcases; days 7–9 run the bounded oracle/passive diagnostic; days 10–12 run local-flow and shuffled-information controls only if the oracle passes; days 13–14 record one of task-infeasible, estimator-infeasible, or policy-feasible.
- **Full workplan:** Preserve SPS-P02 as negative calibration; allow one preregistered task-parameter repair if the oracle fails; rerun the oracle once; if feasible, freeze the passive-adjusted estimator and minimum effect before independent confirmation; report signal and null arms separately so differencing cannot hide endpoint saturation.
- **Kill criterion / dependency / risk:** Kill or redesign the first-interception task if the full-state oracle fails after one bounded repair. Reject the local mechanism if it does not beat stationary and shuffled controls. Depends on diagnostic instrumentation, not on MARL. Principal risks are first-event saturation, wall accumulation, sparse valid tracks, and swept-area confounding.
- **Candidate conference and verified deadline:** This is not a separate AAMAS paper by itself. MODELSWARD 2027 could fit a validated evaluation methodology; its official regular-paper deadline is **15 September 2026** (`https://modelsward.scitevents.org/CallForPapers.aspx?y=2027`, checked 2026-07-31). AAMAS 2027 is conditional on FR-009 also succeeding.
- **Status / priority:** `unvalidated`, 10/10 validity gate; no independent paper authorization.

## SPS-FR-011 — Team size and signal-value dilution

- **Provenance:** SPS-P02 `baseline_summary.json`.
- **Motivating exploratory result:** At rho=2 on 12 calibration seeds, single-collector mean gain was 0.06938 versus 0.01104 for four independent collectors; this is not a causal team-size comparison.
- **Question:** After matching passive capture hazard, swept area, sensing, and total action budget, how does collector count change incremental signal value?
- **Falsifiable hypothesis:** Apparent dilution either persists under matched resources or disappears once first-event saturation and passive hazard are matched.
- **Possible contribution:** Separation of parallel-search opportunity from information value and collector interaction.
- **Workplan:** preregister M in `{1,2,4,8}`; match passive hazard and swept area; evaluate absolute event time, paired gain, survival, and ownership on independent seed blocks.
- **Kill criterion:** the M effect vanishes under matching or is an initialization/censoring artifact.
- **Candidate venue:** AAMAS only if interaction survives matching; otherwise robotics/stochastic systems.
- **Status / priority:** unvalidated, 8/10.

## SPS-FR-012 — Calibrating simultaneous inference for censored paired events

- **Provenance:** `results/derived/inference_calibration*.json` and SPS-P02 `primary_analysis.json`.
- **Motivating limitation:** Studentized max-bootstrap null calibration passed two synthetic families, but power, boundary-location error, nonmonotone alternatives, and realistic censoring remain uncalibrated; the 12-seed pilot critical value was 6.187.
- **Question:** Which simultaneous procedure best controls false grid crossings while retaining power under paired covariance, zero inflation, censoring, and nonmonotone curves?
- **Possible contribution:** Reusable calibration protocol for grid-censored simulator boundaries.
- **Workplan:** compare max-bootstrap, paired max-T permutation, and simultaneous-t controls over synthetic and pilot-fitted held-out families; measure coverage, power, false crossing location, and censoring error.
- **Kill criterion:** no robust advantage over a simpler calibrated procedure or benchmark-specific findings only.
- **Candidate venue:** simulation methodology or statistical computing; AAMAS only as a reusable evaluation standard.
- **Status / priority:** unvalidated, 7/10.

## SPS-FR-013 — Coupled numerical endpoint audit

- **Provenance:** SPS-P01; SPS-P02 exact-contact replication; SPS-WO-04 Gate 3; SPS-WO-06 `convergence_report.json` (2026-08-01); 2026-07-31 pre-repair fresh AAMAS review.
- **Motivating observation:** Exact piecewise-specular within-step contact changed 0 of 144 SPS-P02 outcomes, but agreement between two discrete implementations did not establish timestep stability. The active endpoint was subsequently changed from first interception to fixed-window unique capture yield.
- **Research question:** Does the shared-minus-independent first-interception contrast remain stable under coupled-noise refinement from `dt=0.01` to `dt=0.005`?
- **Falsifiable hypothesis:** With finest-grid Brownian increments summed exactly to coarser grids, the mean normalized contrast changes by no more than 0.0025 between the two finest timesteps and no policy ordering reverses.
- **Possible contributions:** A reproducible event-time audit that separates contact-geometry correctness, stochastic time-discretization error, and policy-ranking stability; a reusable provenance format for first-hit simulation benchmarks.
- **Nearest literature / novelty risk:** Gobet (2000), `https://doi.org/10.1016/S0304-4149(99)00109-X`, establishes weak-error concerns for Euler approximation of killed diffusions; Gobet and Menozzi (2010), `https://arxiv.org/abs/0706.4042`, studies boundary corrections and overshoot. A three-level convergence check alone is standard verification, not a publishable numerical method.
- **Minimal experiment / data / compute:** Coupled `dt in {0.02,0.01,0.005}`, fixed physical horizon and `alpha`, 8 diagnostic seeds, stationary plus full-state oracle first, then independent/shared only after their code is frozen; CPU only.
- **Observed validity slice:** The preregistered oracle-minus-stationary gate passed on diagnostic seeds 3001--3008: mean contrasts were 8.625 particles at `dt=0.02`, 8.625 at `dt=0.01`, and 8.375 at `dt=0.005`; the primary mean difference was 0.0 and there were 0/8 seed-level sign changes. The finest-level difference of 0.25 particles was informational. This validates only the oracle/stationary fixed-yield slice; the stated shared-minus-independent hypothesis and normalized 0.0025 tolerance remain untested.
- **Two-week validation plan:** Days 1–3 unit-test exact aggregation of fine Brownian increments; days 4–6 validate deterministic contact microcases across timesteps; days 7–10 run the two-policy slice; days 11–12 add independent/shared if upstream gates pass; days 13–14 quantify disagreement, interval width, and whether a first-passage correction is needed.
- **Full workplan:** Preserve per-level inputs and checksums; compare event ownership, censoring, absolute restricted means, passive-adjusted effects, and policy contrasts; if unstable, test one preregistered correction or finer level before any scientific inference; publish the audit as supplementary methodology unless it reveals a broadly reusable failure mode.
- **Kill criterion / dependency / risk:** Retire as a standalone direction if standard refinement is sufficient and no generalizable issue appears. Block the active scientific claim if the 0.0025 stability tolerance or policy-order condition fails. Depends on a passing oracle microcase. Risk: rare events may make the diagnostic interval too wide even when mean bias is small.
- **Candidate conference and verified deadline:** MODELSWARD 2027 is a conditional methodological venue with a **15 September 2026** regular-paper deadline (`https://modelsward.scitevents.org/CallForPapers.aspx?y=2027`, checked 2026-07-31); otherwise this belongs in the active paper’s reproducibility supplement, not a separate submission.
- **Status / priority:** partially validated validity audit; active shared/independent extension remains untested; 8/10 as a validity audit and 3/10 as a separate paper.
