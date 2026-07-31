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

## SPS-FR-008 — Scale transfer

- **Provenance:** canonical-task generalization limitation.
- **Question:** Which dimensionless groups permit transfer across arena scale, sensing radius, particle density, and diffusion?
- **Falsifiable hypothesis:** matched dimensionless groups collapse interception curves within a frozen tolerance.
- **Possible contribution:** Similarity laws and out-of-distribution benchmark splits.
- **Novelty threat:** known dimensional analysis may fully determine the result.
- **Kill criterion:** no curve collapse or no result beyond standard nondimensionalization.
- **Candidate venue:** complex systems or simulation; verify current venue details.
- **Workplan / status:** derive groups, preregister collapse error, run bounded factorial pilot; inactive.

## SPS-FR-009 — Bounded evidence fusion as the actual multi-agent question

- **Provenance:** SPS-P01 and the 2026-07-31 Research Lead AAMAS relevance audit.
- **Motivating result:** `local_flow_v1` ignores teammate information, so the nominal team is exactly four independent controllers; no coordination claim is identifiable.
- **Question:** At fixed collector count, sensing, action, parameter, and tuning budgets, does a bounded shared velocity-summary lower the grid-censored first-interception boundary relative to four independent local-flow collectors?
- **Falsifiable hypothesis:** A prespecified low-bandwidth summary produces a positive paired boundary shift after passive-flux and swept-area controls.
- **Possible contribution:** A causal value-of-information result near weak stochastic structure, rather than a duplicated-collector benchmark.
- **Minimal experiment:** freeze a two-number robust team-flow summary and a communication dropout control; compare against the unchanged `local_flow_v1` under identical seeds and capacity.
- **Kill criterion:** no boundary shift, leakage through the message, or advantage explained by policy capacity or centralization.
- **Candidate venue:** AAMAS only if the mechanism survives all matched-budget controls; otherwise keep it within a benchmark or stochastic-control paper.
- **Status / priority:** `unvalidated`; 10/10 because it is the clearest path to genuine multi-agent relevance.

## SPS-FR-010 — Policy-specific signal value beyond passive transport

- **Provenance:** SPS-P02 `baseline_summary.json` and `primary_analysis.json`.
- **Motivating exploratory result:** At rho=2 on 12 calibration seeds, mean gain was 0.01104 for local flow versus 0.01313 stationary and 0.01292 for both random and privileged oracle; these are non-confirmatory descriptions.
- **Question:** Does a causal local policy produce signal-induced first-interception gain beyond passive field transport?
- **Falsifiable hypothesis:** A preregistered difference-in-differences, `E[D_policy(rho)-D_stationary(rho)]`, is positive over a persistent grid region and survives velocity/history shuffles and field-orientation controls.
- **Possible contribution:** Causal decomposition of passive transport, local evidence use, and motion policy.
- **Workplan:** freeze the contrast before new seeds; include stationary, action-shuffled, velocity-shuffled, density, coverage, and oracle controls; report event-time survival curves; stratify orientation against the lattice.
- **Kill criterion:** no oracle headroom or all advantage is passive flux, lattice orientation, or first-event saturation.
- **Candidate venue:** stochastic control, robotics, or complex systems; AAMAS only with a surviving coordination mechanism.
- **Status / priority:** unvalidated, 10/10; required before training.

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
