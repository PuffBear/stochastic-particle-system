# SPS-WO-04 Literature Scout Handoff

**Date:** 2026-07-31

**Role:** Literature Scout

**Scope:** Focused repair after the passive-baseline calibration result

**Files changed:** `paper/literature.md`, this handoff only

**Workspace check:** passed with the research-program checker before editing

## Executive conclusion

The literature supports the experimental correction, but narrows the novelty claim sharply. Stationary capture in chaotic flow is a published mechanism, so the primary causal contrast must be active policy improvement beyond passive advection. Low-bandwidth collective sensing and learned communication are established families, so a fixed team-mean velocity message is a diagnostic mechanism rather than a novel communication algorithm. Finally, exact within-step segment–disc contact does not establish numerical convergence of a stochastic first-contact outcome; a coupled dyadic timestep study is mandatory.

## Verified additions

| Topic | Source | Date | Inspected status | Actionable implication |
|---|---|---:|---|---|
| Passive capture | Wang et al., “Particle capture in a model chaotic flow,” https://doi.org/10.1103/PhysRevE.104.064203 | 2021-12; erratum 2024-09 | Abstract/bibliographic page | Treat stationary signal/null as mandatory comparator and report active-minus-stationary difference-in-differences. |
| Mobile capture | Wang et al., “Mobile-collector capture of particles in a chaotic flow,” https://pmc.ncbi.nlm.nih.gov/articles/PMC12331103/ | 2025-08-07 | Relevant full-text sections inspected | Decompose lab-frame motion into passive flow advection plus controlled relative motion; avoid regimes where strategy differences collapse. |
| Distributed sensing in flows | Shaffer et al., https://arxiv.org/abs/2509.14228 | 2025-09-17 v1 | Abstract/version page | Do not claim first distributed sensing framework in complex flows; distinguish passive-particle velocity inference and capture from model-based source localization. |
| Collective sensing | Hornischer et al., https://arxiv.org/abs/1903.05444 | 2019-03-13 v1 | Abstract/version page | Generic local-communication collective sensing is not novel. |
| Learned communication | Foerster et al., https://proceedings.neurips.cc/paper_files/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html | NeurIPS 2016 | Official abstract | A fixed message is a mechanism probe; learned-communication claims require a learned comparator. |
| Bandwidth-efficient MARL | Wang et al. (IMAC), https://proceedings.mlr.press/v119/wang20i.html | ICML 2020 | Official abstract | State exact budget; compare no communication and budget-matched communication if efficiency is claimed. |
| Event-triggered communication | Hu et al., https://arxiv.org/abs/2010.04978 | 2020-10-10 v1 | Abstract/version page | Required only if the scope expands to communication efficiency or scheduling. |
| Discrete hitting bias | Gobet, https://doi.org/10.1016/S0304-4149(99)00109-X | 2000-06 | Publisher abstract | First-contact metrics require timestep convergence; related discrete killing has exact order O(sqrt(Δt)) weak bias under stated assumptions. |
| Boundary correction | Gobet and Menozzi, https://arxiv.org/abs/0706.4042 | 2007 preprint / 2010 journal | Abstract/version page | Boundary-shift correction is a fallback robustness method, not the first test. |
| Coupled levels | Giles, https://doi.org/10.1287/opre.1070.0496 | 2008 | Institutional/author abstract | Couple Brownian paths across levels; do not compare independent timestep runs. |
| Adaptive Brownian path | Jelinčič et al., https://arxiv.org/abs/2405.06464 | v6 2025-09-16 | Abstract/version page | If contact refinement becomes adaptive, query one reproducible Brownian path instead of redrawing noise. |

No claim above is based on a claimed full-paper read except the explicitly identified relevant sections of the open-access 2025 Wang et al. article.

## Mandatory experiment changes

1. Add signal/null stationary outcomes and make the primary mechanism diagnostic a policy-minus-stationary difference-in-differences.
2. Add a privileged full-state first-interception oracle and privileged true-local-velocity policy. These distinguish task failure, controller failure, and estimator failure.
3. For the proposed bounded message, compare no message, exact team summary, shuffled/permuted summary, and delayed summary. Add quantization/noise only if claiming bandwidth efficiency.
4. Keep one-versus-four independent-versus-four shared as separate deployment contrasts.
5. Run a small dyadic Δt, Δt/2, Δt/4 pilot with Brownian increments coupled by exact summation and stable event identities.
6. Do not rerun the 24-seed broad sweep before these diagnostics establish a plausible effect and simulation-based power target.

## Decision tree for the Research Lead and Engineer

- **Oracle fails to beat stationary:** stop policy tuning; repair task, outcome, horizon, speed budget, or geometry.
- **Oracle works; true-local-velocity fails:** repair action semantics/controller.
- **True-local-velocity works; estimated local flow fails:** repair observation history, velocity estimator, wall/contact contamination, and action timing.
- **Estimated local flow works; team summary does not beat independent team:** retain as flow-control benchmark, not an AAMAS coordination claim.
- **Team summary works but shuffled/delayed summary also works:** investigate regularization or action-synchronization explanations; do not attribute gain to shared evidence.
- **Effect changes materially across timestep levels:** withhold scientific interpretation until discretization is repaired or extrapolated.

## Novelty wording allowed now

Safe provisional wording:

> We study whether a fixed-dimensional summary of locally estimated passive-particle motion changes the policy-specific weak-signal exploitation boundary for a team of mobile collectors, beyond gains explained by passive advection and independent parallel search.

Unsafe wording:

- first mobile particle collector;
- first distributed sensing method in a flow;
- first low-bandwidth MARL communication method;
- universal detectability boundary; or
- timestep-exact first-contact statistics.

## Remaining literature gaps

These are useful but should not block the next bounded diagnostic:

1. Inspect the full Shaffer et al. preprint before any related-work prose compares algorithms or experimental details.
2. Retrieve the full 2021 Wang et al. stationary-collector paper and its 2024 erratum before implementing a faithful stationary double-gyre analogue.
3. Search specifically for moving absorbing traps/targets under reflected diffusion if timestep bias is material in the pilot.
4. Search recent AAMAS/CoRL/RSS work on communication ablations tied to causal message usefulness before promoting the fixed summary into a learned communication paper.

## Provenance note

Search performed 2026-07-31 against current primary/official records. Publication/version dates and access states are recorded in `paper/literature.md`. No experiment, code, result, manuscript, or claim ledger file was modified by this role.
