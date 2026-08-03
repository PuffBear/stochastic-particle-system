# FR-B3 Venue Assessment and Roadmap

**Assessment date:** 3 August 2026

## Venue decision

### Primary: ICLR 2027

ICLR 2027 is the clear A* venue in the requested deadline window. Its official
dates are:

- abstract: 18 September 2026, Anywhere on Earth;
- paper: 25 September 2026, Anywhere on Earth.

The call explicitly includes reinforcement learning, datasets and benchmarks,
general machine learning, robotics/autonomy, and physical-science applications.
ICLR does not require every paper to introduce a neural-network architecture.

Official call: <https://iclr.cc/Conferences/2027/CallForPapers>

### Strong topical fallback: AAMAS 2027

AAMAS is a better topical match for a paper led by multi-agent coordination,
communication, and swarm control. Its official deadlines are:

- author registration: 17 September 2026;
- abstract: 1 October 2026;
- paper: 8 October 2026, Anywhere on Earth.

The 2027 call explicitly lists multiagent learning, learned communication,
swarm and multi-robot behavior, reproducible MAS benchmarks, and coordination.
Under the [current ICORE 2026 listing](https://portal.core.edu.au/conf-ranks/922/)
AAMAS is ranked A, while it was A* in CORE 2023. It is therefore a topical
fallback, not the answer to a strict current A* constraint.

Official call: <https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/>

## Current condition

Before this revision, FR-B3 was a research sketch rather than a submission:

- seven overlapping planning documents;
- no FR-B3 runner, configs, tests, or results;
- a wrong historical anchor (`kappa=0.20` instead of executed `0.50`);
- reversed interpretation of both axes in several places;
- a two-axis sufficiency claim that omitted absolute transport scale;
- selective seed expansion and an arbitrary log offset;
- unsupported real-domain estimates;
- no learned generalization experiment connecting the work to ICLR's core
  audience.

After registration and the passed scale-equivalence audit, the mathematical,
software, and implementation-correctness foundations are in place. A
fail-closed HPC execution package, post-run validator, publication-figure
pipeline, source-backed literature map, learned-transfer candidate protocol,
and results-gated manuscript skeleton are also prepared. The paper remains
early: the registered-seed factorial has not run, the central result is
unknown, and no learned scale-transfer experiment has been completed. Treat
this as roughly 30% of an ICLR submission, not as a nearly finished paper.

## Feasibility

ICLR is feasible but high risk. There are 46 days to the abstract deadline and
53 days to the paper deadline from the assessment date. Submission should
proceed only if both of these gates are met:

1. By 16 August: the full 27-cell scripted factorial is complete, audited, and
   yields a stable scientific result (whether or not two-axis sufficiency is
   rejected).
2. By 31 August: at least one communicating and one non-communicating learned
   policy have a completed raw-versus-nondimensional scale-transfer study.

If the first gate fails, stop the ICLR sprint. If the first passes but the
second fails, the work is much better positioned for AAMAS than ICLR.

## Tasks to an acceptable ICLR submission

### 1. Freeze and run the causal benchmark (3-16 August)

- Advisor-check the three-group derivation, axis ranges, decision margin, and
  seed budget before changing protocol status.
- [x] Run the scale-equivalence audit. Audit v2 passed all eight seed-policy
  comparisons after a versioned physical-unit clipping correction; see
  [`RESCALING_AUDIT.md`](RESCALING_AUDIT.md).
- Run all 27 cells, 64 fresh common seeds, and four policies.
- [x] Prepare a fail-closed PBS package that preflights the exact 6,912-episode
  design and refuses dirty, wrong-branch, or overwrite-prone execution.
- [x] Prepare a post-run completeness/provenance validator and immutable figure
  pipeline for the paired-gain surface and held-out two-axis versus three-axis
  comparison.
- Generate the registered outputs by running the prepared pipeline on HPC.
- Report the result honestly if the original two-axis claim fails.

### 2. Add the ICLR-critical representation test (10-31 August)

The distinctive learning question should be:

> Do learned multi-agent policies generalize between physically rescaled but
> dimensionlessly equivalent environments, and does nondimensionalizing their
> observation representation restore that invariance?

Minimum credible experiment:

- one independent learner (IPPO or MAPPO) and one communicating learner
  (CommNet);
- raw observation representation versus a dimensionless representation;
- train at the canonical scale;
- test without fine-tuning on the three equivalent rescalings;
- in-scale controls, three or more training seeds, learning curves, and final
  policy evaluation on the common FR-B3 seed panel.

This experiment is necessary for strong ICLR fit. It is not needed to validate
the scripted catchability surface itself.

The unregistered, seed-firewalled design is now specified in
[`LEARNED_TRANSFER_PROTOCOL.md`](LEARNED_TRANSFER_PROTOCOL.md). It must not run
on candidate seeds until its remaining tuning, effect-threshold, implementation,
and human sign-off blockers are resolved.

### 3. Robustness and ablations (17 August-5 September)

- denser follow-up points near any apparent boundary;
- alternate horizon or sensing-radius slice to delimit the claim;
- shuffled and leave-self-out message controls already available upstream;
- outcome decomposition: first contact, post-contact conversion, and unique
  capture yield;
- oracle headroom and saturation checks;
- compute/runtime and numerical-equivalence audit.

### 4. Manuscript and review (24 August-25 September)

- Draft results-first; do not retain the old domain-mapping narrative.
- [x] Prepare an anonymous ICLR-style skeleton whose scientific conclusions and
  figures remain visibly blocked on validated outputs.
- [x] Prepare a primary-source literature positioning memo and bibliography.
- Position against dimensionless RL, sim-to-real/domain randomization, learned
  communication, and multi-agent benchmark literature.
- Complete figures and appendix by 10 September.
- Obtain at least two independent technical reviews before the 18 September
  abstract deadline.
- Use the title and abstract to state the result actually observed, not the
  hoped-for two-axis collapse.

## Go/no-go rule

An ICLR submission is justified only if the final paper has both:

1. a decisive, reproducible result about the minimum dimensionless
   parameterization of coordination gain; and
2. a learning result showing why that parameterization matters for policy
   representation or scale generalization.

Without item 2, submit to AAMAS 2027 or continue toward a later ML venue rather
than stretching a scripted 27-cell sweep into an ICLR claim.
