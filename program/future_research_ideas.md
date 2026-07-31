# Future Research Ideas

Inactive directions only; none may consume active-paper resources without a Program Director decision. Each entry is unvalidated.

## SPS-FR-001 — Communication shifts a weak-field boundary

- **Question:** Under identical local observations and policy-capacity budgets, does a bounded shared summary lower the first-interception detectability boundary?
- **Possible contribution:** A causal, matched decomposition of coordination value near a weak-signal transition.
- **Why deferred:** It may become the strongest AAMAS question, but changing the active research question requires user approval.
- **Novelty threats:** Communication can leak global state; MAPPO alone is not a communication mechanism; benefits may be explained by larger joint swept area.
- **Candidate venues:** AAMAS if the multi-agent mechanism is isolated; otherwise an embodied-AI or robotics venue.
- **Workplan:** freeze message budget; compare the same policy class with message ablation; match observation, parameter, action, and collector budgets; estimate paired boundary shifts.

## SPS-FR-002 — Event-keyed counterfactual random numbers

- **Provenance:** SPS-WO-01; SPS-WO-02; `program/handoffs/SPS-WO-02-engineering.md`.
- **Motivating observation:** Particle Brownian forcing and policy actions are now pre-generated, but capture ties still consume a stateful RNG. Divergent episodes could therefore resolve later ties using different random draws.
- **Question:** Which random-number construction best preserves causal matching in interacting particle simulations after trajectories diverge?
- **Falsifiable hypothesis:** Event-keyed tie draws indexed by scenario, step, particle, and eligible-collector set are invariant to unrelated event-consumption order while naïve stateful draws are not.
- **Possible contribution:** A reproducibility and variance-reduction protocol comparing full pre-generation, event-keyed streams, and naïve stateful streams.
- **Why deferred:** It is primarily a methodology paper unless the active benchmark reveals material bias.
- **Minimal experiment:** construct two trajectories with identical tie events but different unrelated prior event counts; verify that event-keyed ownership is identical and stateful ownership can differ.
- **Required compute:** CPU-only unit tests and a bounded Monte Carlo diagnostic.
- **Two-week validation plan:** implement keyed hashing; add collision and permutation tests; quantify variance under a small paired simulator; search the nearest simulation-coupling literature.
- **Candidate venue:** methodology or simulation venue only after a verified material effect; no deadline is asserted without a fresh venue check.
- **Status / priority:** `unvalidated`; 8/10 because it is an active correctness blocker even if it never becomes a paper.

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

- **Question:** How rapidly can a local team adapt when field orientation, centre, or topology changes within an episode?
- **Possible contribution:** Detectability-versus-adaptation phase diagram with bounded memory.

## SPS-FR-005 — Strategic or learned particle evasion

- **Question:** How does the boundary change when particles respond strategically to collectors?
- **Possible contribution:** Genuine pursuit-evasion game extending the passive-particle benchmark.

## SPS-FR-006 — Heterogeneous collector teams

- **Question:** When do heterogeneous sensing and actuation capabilities outperform homogeneous allocation under the same total resource budget?
- **Possible contribution:** Role emergence and capability allocation near weak signals.

## SPS-FR-007 — Offline MARL from paired trajectories

- **Question:** Can paired signal/null trajectory datasets support reliable offline policy selection near a detectability boundary?
- **Possible contribution:** Counterfactual benchmark for offline multi-agent evaluation.

## SPS-FR-008 — Scale transfer

- **Question:** Which dimensionless groups permit transfer across arena scale, sensing radius, particle density, and diffusion?
- **Possible contribution:** Similarity laws and out-of-distribution benchmark splits.
