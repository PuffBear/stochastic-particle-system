# Stochastic Particle Lab — Correction and Yield-Gate Report

**Date:** 31 July 2026  
**Active scope:** Paper A only  
**Repository target:** `PuffBear/stochastic-particle-system`, branch `research-autonomy`  
**Current stage:** fixed-horizon action-headroom gate passed; numerical validation and the actual shared-versus-independent experiment remain unexecuted

## Five-bullet TL;DR

- The original first-interception question failed its decisive corrected-oracle gate. Under SPS-WO-04, the full-state oracle's passive-adjusted means were negative at all three nonzero catchable points, with only 0/8, 1/8, and 2/8 seeds in the favorable direction. This endpoint is retired for the active coordination study.
- The failure was diagnosed as first-event saturation, not insufficient compute: contacts occurred within the first few steps, every episode was uncensored, stationary averaged 5.78 steps to contact across the grid, and the corrected oracle averaged 3.00 steps. The 400-step horizon and walls were not the limiting factors.
- SPS-WO-05 changed only the outcome and stopping rule: count distinct team captures through inclusive step 67 and continue after the first contact. On eight fresh diagnostic seeds at `alpha=0.06`, the full-state oracle beat stationary by 9.375 unique particles on average, with all 8/8 seed differences positive, and beat the true-field-only control by 7.5 on average, with 7/8 positive.
- The engineering package is healthy: all 89 tests pass, the 48-episode canonical R1 run has complete matched-stream provenance, and its principal files match their recorded SHA-256 hashes. SPS-P05 remains permanently excluded, and a duplicate WO-05 execution caused by a process-status race is also excluded rather than counted as replication.
- The positive oracle diagnostic establishes only that legal actions can improve the redesigned capture-yield endpoint. It is **not coordination evidence**. Next, preregister and run coupled-noise timestep validation; only if that passes, run the bounded shared-summary versus capacity-matched independent diagnostic with attribution controls.

## Explain it simply

The earlier experiment asked which policy caused the **first** capture sooner. That sounds natural, but in this environment the first capture often happened almost immediately, sometimes within one or two simulation steps. When an event is already pressed against the earliest possible time, there is almost no room for a better controller to show improvement. Adding more seeds would measure that floor more precisely; it would not repair the question.

The corrected design asks a different, physically motivated question: during the time required for a fastest-moving collector to travel one sensing radius, how many distinct particles can the team capture? That window is 67 steps, or 1.34 physical time units. A capture on step 1 is still recorded, but it no longer ends the episode.

This repair worked as an upstream feasibility test. A controller with full particle positions and legal actions captured meaningfully more particles than standing still or merely knowing the field direction. That tells us the redesigned task gives actions enough time to matter. It does **not** tell us that agents benefit from sharing information. The actual paper question compares a bounded three-number shared velocity summary with an identical-capacity independent controller, and that comparison has not yet been run.

The scientific progression is therefore disciplined: first show that the numerical result is stable when the timestep is refined using coupled noise; then test sharing against an independent controller and controls that separate genuine multi-agent value from ordinary pooled denoising or synchronized motion.

## Verified provenance

The canonical scientific record is `results/raw/SPS-WO-05-YIELD-GATE-R1/`. Its manifest records:

- repository: `PuffBear/stochastic-particle-system`;
- branch: `research-autonomy`;
- base revision: `1289499b4d00c7384c8364cc534f69ba552d87df`;
- Python 3.12.13 and NumPy 2.3.5 on Linux;
- runtime: 190.846 seconds on Codex cloud CPU;
- workload: 48 episodes and 3,216 environment steps;
- conditions: seeds 2001–2008, `alpha` in `{0, 0.06}`, and stationary, true-field-only, and full-state-oracle policies;
- confirmation eligibility: false.

The recorded and independently recomputed hashes agree:

- `episode_summaries.jsonl`: `1ae5a80b06b783ad11b044b48ed37e82fda21e7f0eb52f09daaabe4621349b3c`;
- `capture_events.jsonl`: `6621bc43d4fbd14c7f1f07cacfb5521db999afc90a2fa34d4ab1a966c4e72724`;
- `oracle_gate.json`: `a54d75875b8fbca4a9b1b50e4336389a8de0a12f0cbc88b4fa87ff6ff9581226`.

The unit suite was rerun with `PYTHONPATH=src python -m unittest discover -s tests -q`: **89 tests passed, 0 failed**.

The exact scientific chain is preserved in:

- `program/work_orders/SPS-WO-04.json`;
- `program/handoffs/SPS-WO-04-engineering.md`;
- `results/raw/SPS-P04-ORACLE-STATE-REPAIR/`;
- `program/work_orders/SPS-WO-05.json`;
- `program/handoffs/SPS-WO-05-research-lead.md`;
- `program/handoffs/SPS-WO-05-engineering.md`;
- `results/raw/SPS-WO-05-YIELD-GATE-R1/`;
- `results/derived/SPS-P05-procedural-status.json`;
- `results/derived/SPS-WO-05-execution-race-audit.json`;
- `paper/experiments.jsonl` and `paper/claims.csv`.

The local artifact manifest records that the WO-05 source snapshot includes workspace changes relative to the stated base revision. This report does not infer a newer remote commit from local files alone; repository publication remains separately auditable on the configured branch.

## Paper A

### Status

Active, with one paper only. The action-headroom prerequisite for the redesigned endpoint passed. The central multi-agent claim, SPS-C03, remains blocked because no shared-versus-independent result exists.

### Exact research question

At `alpha=0.06` in the canonical four-collector task, does replacing each collector's independent local velocity estimate with the same bounded three-number team velocity summary increase the number of distinct particles captured through inclusive step 67, under matched observation, action, computation, and policy-capacity budgets?

This is one causal comparison. Oracle validity, timestep stability, message controls, and power are gates for answering it; they are not additional paper questions.

### What failed under SPS-WO-04

SPS-P03 used a flawed oracle input: the last realized Brownian displacement was treated as predictive. The run is preserved as a failed implementation diagnostic. The permitted repair, SPS-P04, replaced that input with the known current deterministic drift and left the physical task unchanged.

The corrected oracle still failed every preregistered first-event feasibility point:

- `alpha=0.03`, `kappa=0.25`: mean passive-adjusted gain `-0.00844`, favorable direction 0/8 seeds;
- `alpha=0.06`, `kappa=0.50`: mean `-0.00563`, favorable direction 1/8;
- `alpha=0.12`, `kappa=1.00`: mean `-0.01063`, favorable direction 2/8.

All diagnostic episodes were uncensored. Across the grid, stationary first contact averaged 5.78 steps and the corrected oracle averaged 3.00 steps; the oracle median was step 2, with a reported range of steps 1–8. No pre-contact wall-proximity event explained the early contacts. The first-event outcome was therefore saturated near its lower bound and could compress the signal-minus-null passive-adjusted contrast even when the oracle was absolutely earlier.

The decision was to kill/redesign the first-interception endpoint, not to buy more compute, tune after seeing outcomes, or proceed to communication experiments.

### WO-05 redesign and result

SPS-WO-05 changed exactly the endpoint and episode stopping rule:

- old: stop at the team's first interception and report its one-based step;
- new: continue through inclusive step 67 and count distinct particles captured by the team.

The window was derived before WO-05 outcomes were inspected:

`ceil(sensing_radius / (collector_max_speed × dt)) = ceil(0.16 / (0.12 × 0.02)) = 67`.

The reset, 256 particles, four collectors, geometry, capture radius, sensing radius, field, diffusion, action limits, policy parameters, and matched stochastic streams remained fixed.

At `alpha=0.06`, the eight seed-level oracle-minus-stationary capture-yield differences were `[11, 13, 9, 12, 7, 7, 11, 5]`. Their mean was **9.375**, median **10**, sample standard deviation **2.825**, descriptive paired-bootstrap 95% interval **[7.5, 11.25]**, and direction was positive in **8/8** seeds.

The oracle-minus-true-field-only differences were `[6, 12, 6, 15, 8, 3, 11, -1]`. Their mean was **7.5**, median **7**, sample standard deviation **5.155**, descriptive interval **[4.125, 10.875]**, and direction was positive in **7/8** seeds.

Every frozen gate passed: execution and correctness; a mean oracle-minus-stationary gain of at least four; at least six favorable seed directions; a positive mean targeting contrast; matched streams; checksums; and complete artifacts. These intervals are descriptive because all eight seeds are diagnostic.

### Meaning

The unchanged physical task contains practically relevant **action-contingent capture-yield headroom** when evaluated over the redesigned movement-scale window. Full particle positions plus target-aware legal action can improve yield over passive collection and true-field direction alone.

This result does not show that local observations suffice, that the shared summary works, that communication helps, that collectors coordinate, or that MARL is useful. It authorizes numerical validation and a later sharing diagnostic; it does not support SPS-C03.

### Strongest threat

The paper still does not answer its own question. The positive treatment is a privileged full-state target-assignment oracle, while the proposed decentralized shared-summary treatment remains unexecuted. Even if sharing later helps, a spatially uniform field makes ordinary pooled denoising or common synchronized motion a strong mundane explanation. The selected 67-step horizon also creates an adaptive-design concern that must be addressed with a preregistered robustness range and seed separation.

### Confidence

- High confidence that the recorded WO-05 diagnostic executed as specified and passed its frozen engineering and descriptive continuation gates.
- Moderate confidence that first-event saturation was the main reason the prior endpoint was uninformative; the evidence is strong but does not uniquely eliminate initialization geometry, capture radius, or oracle quality as contributors.
- No confidence claim is warranted for the shared-versus-independent effect because it has not been measured.

## Paper B

Paper B is intentionally inactive. No second question has been authorized, and no work is being diverted into a second paper. A second paper may activate only if its question, claims, experiments, and novelty threats become genuinely separable from Paper A.

## Engineering

The correction cycle delivered a distinct full-state action-feasible oracle, detailed per-step diagnostics, exact coupled-Brownian aggregation utilities, bounded permutation-invariant team summaries, capacity-matched independent/shared controller shapes, and the fixed-horizon unique-yield runner.

The latest deterministic tests cover unique counting, inclusion of step 67, exclusion of step 68, continuing after first contact, matched null/signal provenance, action bounds, observation leakage, permutation invariance, agent-action equivariance, exact Brownian aggregation, exact piecewise-specular contact, event-keyed ties, schema validation, and immutable artifact handling. The full suite passes 89/89.

SPS-P05 is permanently excluded. It changed the particle count after SPS-WO-04's single-repair allowance had already been consumed. Its files remain immutable for audit, but its outcomes may not enter claims, design selection, manuscript decisions, or power calculations.

The WO-05 execution-race audit found a separate process-control error. The first process was mistakenly classified as ended when its tool session returned before the underlying job reached a terminal state. It completed 54 seconds later, after an identical R1 process had begun. The two output files are bitwise identical. R1 is the sole canonical evidence package; the original is excluded and is not pooled, called a replication, or used to improve precision. The process correction is explicit: poll long-running jobs to a terminal process state before classifying or relaunching them.

No HPC or GPU was used. No shared-summary, timestep-science, power, learned-policy, MARL, or confirmatory run was smuggled past the dependency gates.

## Manuscript

`paper/manuscript/main.tex` and the compiled seven-page `paper/manuscript/main.pdf` now preserve the negative SPS-P02 history, the SPS-WO-04 oracle failure, the reason for the endpoint redesign, the WO-05 diagnostic result, the SPS-P05 exclusion, and the WO-05 duplicate exclusion. The abstract and result sections explicitly state that no coordination, timestep-convergence, power, confirmation, learning, or MARL result is claimed.

The manuscript remains an internal progress package, not a submission candidate. It uses a generic `article` layout, contains visible `PENDING` instructions, has a small bibliography, and ends with an unexecuted work plan. These defects should not be cosmetically hidden before the central experiment exists. The result-first progression is: numerical-validity result, shared-versus-independent diagnostic, attribution controls, powered confirmation, then submission-format conversion and narrative compression.

## Literature

No new source was added solely because the WO-05 oracle gate was positive. The most important existing nearest neighbor remains Wang et al. (2025), “Mobile-collector capture of particles in a chaotic flow,” which already studies a locally informed mobile collector and identifies multiple collectors and coordination as future work: https://doi.org/10.1371/journal.pone.0329766.

The strongest AAMAS novelty threats remain:

- Löffler et al. (2023) already demonstrate locally observing active-particle collective foraging with shared PPO: https://doi.org/10.1038/s41598-023-44268-3;
- Atanasov et al. (2015) already study distributed stochastic source seeking with mobile robot networks: https://doi.org/10.1115/1.4027892;
- IPPO and MAPPO are established cooperative multi-agent baselines, so they become relevant only if the program ultimately makes a learned-policy or MARL claim.

The narrow defensible contribution is not “multiple collectors” or “sharing improves sensing.” It would be a causally isolated, capacity-matched result showing when a specific bounded shared statistic improves capture yield beyond independent local estimation, pooled denoising, passive transport, and synchronized common action. That result does not yet exist.

## Fresh AAMAS review

The immutable post-WO-05 review is `paper/reviews/2026-07-31-post-wo05-fresh-aamas.md`.

- Verdict: **Reject**.
- Overall score: **2/10**.
- Reviewer confidence: **5/5**.

The strongest rejection argument is decisive: the paper asks whether sharing a team statistic improves performance, but that treatment has never been run. The positive oracle diagnostic establishes task headroom, not communication or coordination value. The present package is a well-documented pre-experiment rather than a completed AAMAS paper.

The strongest acceptance path is methodological: the matched-counterfactual design, exact event semantics, transparent negative results, and oracle-first task validation could support a useful benchmark paper if accompanied by a complete, public, reproducible multi-agent information-sharing study.

Minimum repairs that could change the review are: pass coupled-noise timestep validation; execute a separately preregistered and adequately powered shared-versus-independent comparison; isolate sharing from equal-effective-sample denoising and common motion; add message-shuffled, delayed/corrupted, teammate, assignment, team-size, horizon, sensing, signal/noise, and timestep controls; provide the public artifact; and rewrite the manuscript around completed evidence.

## Expansionist update

No new research or industry entry was promoted on the basis of the oracle diagnostic alone. That restraint is appropriate because an upstream feasibility result does not establish a paper contribution or product demand.

The active research-backlog connection is `SPS-FR-009`, bounded evidence fusion as the actual multi-agent question. Its scientific premise survives because WO-05 now shows action headroom, but its first-interception wording and thresholds must be revised to the fixed-horizon unique-yield endpoint before reuse. `SPS-FR-013`, the coupled numerical event-time audit, remains the immediate validity dependency, likewise requiring its endpoint language to be updated from first-contact stability to capture-yield stability.

The closest industry hypothesis is `SPS-FI-001`, a distributed-sensing “coordination microscope.” WO-05 strengthens only one component of that idea: the diagnostic workflow can distinguish an uninformative endpoint from an action-sensitive one. It does not validate customer demand, commercial readiness, pricing, or a communication benefit. All industry figures remain explicit low-confidence assumptions, not market evidence or forecasts.

## Blockers and access

No HPC access is needed. The current tasks are small CPU diagnostics and validity checks. No GPU, paid data, credentials, VPN, licensed dataset, or external user decision blocks the next work.

The scientific blockers are:

- timestep stability of the step-67 yield result has not been tested;
- the shared-versus-independent experiment has not been preregistered for the redesigned endpoint;
- the minimum relevant coordination effect, seed budget, and inference plan are not frozen;
- attribution controls must separate shared-information value from pooled denoising, synchronized actions, passive flux, target assignment, and duplicated pursuit;
- the endpoint-development seeds are diagnostic and permanently ineligible for confirmation.

## Autonomous next 24 hours

1. **Preregister the coupled-noise timestep validation before running it.** Keep the physical duration at 1.34, the canonical reset and geometry, and `alpha=0.06`. Generate Brownian increments at the finest timestep and obtain coarser increments by exact summation. Freeze timestep levels, diagnostic seeds, policies, event/yield metrics, policy-order rule, numerical tolerance, uncertainty summary, stop rule, compute cap, and positive/null/reversed/invalid interpretations before viewing outcomes.

2. **Execute the smallest valid timestep slice on Codex CPU.** Start with stationary, true-field-only control, and the full-state oracle. Preserve per-level initial states, coupled increments, capture events, unique yields, ownership, checksums, commands, runtime, and disagreement diagnostics. A failure blocks the communication experiment and triggers one bounded numerical diagnosis; it does not justify more seeds or a favorable reinterpretation.

3. **If and only if timestep validation passes, preregister the shared-versus-independent diagnostic.** The primary contrast will compare the bounded three-number shared team-velocity summary with the identical-shape, capacity-matched independent controller on the step-67 unique-yield endpoint. Freeze a fresh diagnostic seed set, minimum relevant effect, descriptive gate, and confirmation firewall.

4. **Include attribution controls in that diagnostic.** At minimum: stationary; pregenerated random; coverage; true-field-only; full-state oracle; shared summary; capacity-matched independent; message-shuffled; delayed/corrupted or bandwidth-matched messaging; a centralized pooled-estimation but non-coordinating control; an independent estimator with matched effective sample quality; teammate-position ablation; and target-deconfliction/assignment controls. These controls determine whether any gain is genuinely multi-agent or only denoising/common motion.

5. **Update ledgers before prose.** Record timestep and sharing outcomes in `paper/experiments.jsonl`, keep SPS-C03 blocked until supported by valid evidence, update the decision log, repair FR-009/FR-013 endpoint wording, and only then revise tables, figures, and manuscript claims. Preserve a new immutable fresh AAMAS review after the paper package changes.

6. **Do not start confirmation or MARL training.** A confirmation seed count requires simulation-based type-I-error and power calibration after the diagnostic effect and variance are known. IPPO/MAPPO are not authorized unless a genuine learned-policy question survives the scripted mechanism and attribution gates.

## Optional question and no-reply action

No user question is needed today. Under the no-reply default, work continues with the preregistered coupled-noise timestep validation; only a passing numerical gate permits the bounded shared-versus-independent diagnostic. Paper B, confirmation, MARL training, and HPC remain inactive.
