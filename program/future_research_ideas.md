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
- **Dependency boundary:** Attached growth is not part of SPS-C04. Revisit it only after the fixed-geometry correlation-scale result closes; any later test must ask whether growth changes that boundary under area/perimeter-matched controls rather than silently broadening the active paper.

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

## SPS-FR-009 — Post-boundary localized fusion (downstream of active SPS-C04)

- **Provenance:** SPS-P01; SPS-P02; SPS-WO-04; SPS-WO-07 `attribution_gate.json` (2026-08-01); 2026-07-31 pre-repair fresh AAMAS review; `paper/literature.md` focused repair.
- **Motivating observation:** `local_flow_v1` ignores teammate information, so four collectors are independent replicas. SPS-WO-07 tested one three-number message: clipped team-mean local velocity (two values) plus team validity fraction (one value).
- **Research question:** If the active SPS-C04 phase-boundary study first establishes that all-to-all averaging fails below a correlation-scale threshold, can a covariance-localized three-scalar fusion rule remove that low-correlation loss without sacrificing high-correlation denoising benefit?
- **Falsifiable hypothesis:** Inactive follow-on only. It becomes eligible only if SPS-C04 supports one ordered unique-yield crossover and the predicted estimator-error/action-diversity mechanism. Otherwise there is no identified failure mode for the method to repair.
- **Possible contributions:** A theory-derived mitigation of a demonstrated communication failure mode; a capacity-, observation-, action-, and arithmetic-budget-matched comparison against independent and all-to-all execution.
- **Nearest literature / novelty risk:** Zhang, Martinez, and Masson (2015), `https://doi.org/10.3389/frobt.2015.00012`, already compare independent and shared-information search in a particle-cue environment with explicit correlation length. Taylor et al. (AAMAS 2010), `https://www.ifaamas.org/Proceedings/aamas2010/pdf/01%20Full%20Papers/02_01_FP_0026.pdf`, show topology-dependent reversals where more teamwork hurts under uncertainty without communication cost; Aust et al. (2022), `https://doi.org/10.1007/978-3-031-20176-9_19`, show benefits from limited communication under correlated observations; and Sung et al. (2020), `https://doi.org/10.1007/s10514-019-09856-1`, optimize a non-additive unique-target objective under limited communication. Localized fusion, correlation-aware weighting, harmful sharing, limited communication, and unique-target accounting are not novel. This direction survives only as a prospectively derived repair of the exact SPS-C04 crossover under matched message, action, and arithmetic budgets.
- **Minimal experiment / data / compute:** No experiment is authorized. If activated later, use fresh seeds and representative low/near/high SPS-C04 `eta` conditions; compare independent, all-to-all, and one prospectively derived localized rule under the same three-scalar budget; Codex CPU first.
- **Observed diagnostic:** The frozen three-number summary failed its joint gate. Shared minus capacity-matched independent yield averaged 1.75 particles (descriptive paired-bootstrap 95% interval `[-0.375,3.75]`) and was strictly positive in 4/8 seeds, below the preregistered two-particle and 5/8 thresholds. Seed effects were `[3,6,4,-3,0,0,5,-1]`. The oracle exceeded shared by 7.5 particles on average and in 8/8 seeds, so task/action headroom remains, but the present summary did not reliably exploit it. Heterogeneous signs indicate instability, not established latent regimes.
- **Two-week validation plan:** Deferred. First complete SPS-WO-09 and the fresh SPS-C04 diagnostic. If the phase law survives, derive exactly one localized rule before inspecting new outcomes, unit-test graph and bandwidth equivalence, then preregister its own seed block.
- **Full workplan:** Never use WO-07 or SPS-C04 diagnostic seeds for method selection and confirmation in the same claim. Compare against Nakamura et al. (2022), Elwin et al. (2020), and targeted/graph communication baselines before claiming method novelty.
- **Kill criterion / dependency / risk:** Kill this method direction if SPS-C04 has no ordered crossover, if a privileged localized diagnostic does not repair the low-correlation regime deterministically, if a closest-work adaptation matches the rule, or if the rule lacks a separable causal prediction beyond established limited-communication mechanisms. Main risk is rediscovering established kernel/Voronoi weighting.
- **Candidate conference and verified deadline:** AAMAS 2027 is the intended fit only for a surviving genuinely multi-agent result; the official call lists abstract **1 October 2026** and paper **8 October 2026** deadlines (`https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/`, checked 2026-07-31).
- **Status / priority:** `unvalidated`, inactive and explicitly downstream of SPS-C04; no method implementation, seeds, confirmation, or MARL activation; 3/10 conditional priority after the SPS-WO-11 threat audit.

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

- **Provenance:** SPS-P01; SPS-P02 exact-contact replication; SPS-WO-04 Gate 3; SPS-WO-06 `convergence_report.json` (2026-08-01); SPS-WO-11 endpoint/nonlinear contract; 2026-07-31 pre-repair fresh AAMAS review.
- **Motivating observation:** Exact piecewise-specular within-step contact changed 0 of 144 SPS-P02 outcomes, but agreement between two discrete implementations did not establish timestep stability. The active endpoint was subsequently changed from first interception to fixed-window unique capture yield.
- **Endpoint update:** SPS-WO-11 prospectively freezes `T=1.34=67(0.02)`, corresponding to one sensing-radius traversal. The canonical `400(0.02)=8.0` configuration remains a generic simulator default, not the paper endpoint.
- **Research question:** At fixed `T=1.34`, does the shared-minus-independent distinct-capture contrast remain stable under exactly coupled timestep refinement, and do the frozen analytic-eligibility classifications remain adequate for the mechanism claim?
- **Falsifiable hypothesis:** To be frozen before execution. SPS-WO-11 deliberately does not invent a replacement numerical-equivalence tolerance for the obsolete first-interception threshold.
- **Possible contributions:** A reproducible event-time audit that separates contact-geometry correctness, stochastic time-discretization error, and policy-ranking stability; a reusable provenance format for first-hit simulation benchmarks.
- **Nearest literature / novelty risk:** Gobet (2000), `https://doi.org/10.1016/S0304-4149(99)00109-X`, establishes weak-error concerns for Euler approximation of killed diffusions; Gobet and Menozzi (2010), `https://arxiv.org/abs/0706.4042`, studies boundary corrections and overshoot. A three-level convergence check alone is standard verification, not a publishable numerical method.
- **Minimal experiment / data / compute:** Use `(dt,horizon)` in `{(0.02,67),(0.01,134),(0.005,268)}`, with the same finest Brownian path aggregated upward and the same frozen field realization. Run independent/shared only after the full protocol is frozen; CPU only.
- **Mechanism coverage gate:** Require analytic eligibility at least 80% in every arm--eta cell, between-arm difference at most 10 percentage points, no seed-arm below 50%, clipping at most 1%, and own-empty, fallback, and reflected-valid incidence at most 5%. Collector reflection is capped at 5% only for displacement mediation; rescue and cancellation are uncapped. Failure preserves all-row yield but kills the covariance-supported mechanism. Never discard rows or add seeds post hoc.
- **Observed validity slice:** The preregistered oracle-minus-stationary gate passed on diagnostic seeds 3001--3008: mean contrasts were 8.625 particles at `dt=0.02`, 8.625 at `dt=0.01`, and 8.375 at `dt=0.005`; the primary mean difference was 0.0 and there were 0/8 seed-level sign changes. The finest-level difference of 0.25 particles was informational. This validates only the oracle/stationary fixed-yield slice; the stated shared-minus-independent hypothesis and normalized 0.0025 tolerance remain untested.
- **Two-week validation plan:** Days 1–3 unit-test exact aggregation of fine Brownian increments; days 4–6 validate deterministic contact microcases across timesteps; days 7–10 run the two-policy slice; days 11–12 add independent/shared if upstream gates pass; days 13–14 quantify disagreement, interval width, and whether a first-passage correction is needed.
- **Full workplan:** Preserve per-level inputs and checksums; compare event ownership, censoring, absolute restricted means, passive-adjusted effects, and policy contrasts; if unstable, test one preregistered correction or finer level before any scientific inference; publish the audit as supplementary methodology unless it reveals a broadly reusable failure mode.
- **Kill criterion / dependency / risk:** Retire as a standalone direction if standard refinement is sufficient and no generalizable issue appears. Block the active mechanism claim if the prospectively frozen timestep gate or nonlinear-coverage gate fails. Risk: rare events may make the diagnostic interval too wide even when mean bias is small.
- **Candidate conference and verified deadline:** MODELSWARD 2027 is a conditional methodological venue with a **15 September 2026** regular-paper deadline (`https://modelsward.scitevents.org/CallForPapers.aspx?y=2027`, checked 2026-07-31); otherwise this belongs in the active paper’s reproducibility supplement, not a separate submission.
- **Status / priority:** partially validated validity audit; active shared/independent extension remains untested; 8/10 as a validity audit and 3/10 as a separate paper.

## SPS-FR-014 — Action-diversity mechanism behind harmful global sharing

- **Provenance:** SPS-WO-07 negative diagnostic; active SPS-C04 correlation-scale redesign.
- **Motivating observation:** Global averaging can improve estimation while synchronizing collectors. WO-07's heterogeneous seed effects do not establish that action homogenization caused any loss.
- **Research question:** Within a correlation-scale regime where global averaging underperforms independent estimates, does restoring action diversity while holding the shared message fixed recover fixed-horizon unique capture yield?
- **Falsifiable hypothesis:** A preregistered diversity-preserving action map exceeds the unchanged shared controller on fresh seeds after matching message, action magnitude, path length, and swept area; recovery must co-occur with lower trajectory overlap rather than extra motion.
- **Possible contributions:** Causal separation of estimator benefit from action-correlation cost under an unchanged shared message.
- **Nearest literature / novelty risk:** Piro et al. (2025), `https://arxiv.org/abs/2504.11291`, directly show that spatially correlated detections can create clustering and redundancy while policy heterogeneity restores movement diversity and coverage. Taylor et al. (AAMAS 2010) already show that greater teamwork can hurt under uncertainty. The only potentially separable contribution is a causal intervention that holds the shared estimator and message fixed while changing action diversity, not the observation that diversity can help.
- **Minimal experiment / compute:** Deterministic mechanism microcases, then a bounded CPU diagnostic in one SPS-C04-frozen harmful regime; compare shared, shared-plus-diversity, sham-intervention, and independent policies; record pairwise action cosine, path overlap, unique captures, and duplicate pursuit.
- **Two-week validation plan:** Days 1--3 formalize the intervention and matching; 4--6 microcases and invariance tests; 7--9 bounded diagnostic; 10--12 mundane-explanation controls; 13--14 fresh-review decision.
- **Full workplan:** Activate only after SPS-C04 establishes a stable harmful-sharing regime; freeze one intervention before fresh seeds; use a separate held-out confirmation only if attribution gates pass.
- **Kill criterion / dependency / risk:** Kill if recovery vanishes under motion/coverage matching, overlap does not change, the SPS-C04 regime is not reproducible, recovery is explained by generic coverage or policy heterogeneity including Piro et al.'s mechanism, or no preregistered estimator-side ordered prediction accompanies it. Never merge this intervention into the active phase-boundary question.
- **Candidate venue:** AAMAS only if the causal multi-agent mechanism survives; otherwise a robotics or simulation venue, with its deadline checked at activation.
- **Status / priority:** `unvalidated`, inactive, 3/10 after the SPS-WO-11 threat audit.
