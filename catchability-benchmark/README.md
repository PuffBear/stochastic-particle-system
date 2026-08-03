# FR-B3: Dimensionless Catchability and Coordination Gain

## Research question

> Are per-observation drift SNR (`rho`) and relative drift-to-collector speed
> (`kappa`) sufficient to predict the causal gain from a bounded team velocity
> summary, or is an absolute transport-scale term (`eta`) required?

The primary outcome is the matched-seed difference in unique team capture
yield between `shared_summary_v2` and `capacity_matched_independent`. Both
controllers have the same three-number message capacity; only the shared arm
pools information across collectors.

## Correct dimensionless quantities

For arena characteristic length `L`:

- `rho = alpha * sqrt(dt) / sigma`: drift-to-diffusion ratio in one observed
  displacement. Larger `rho` means the field direction is easier to estimate.
- `kappa = alpha / v_max`: drift speed relative to collector speed. Larger
  `kappa` means lower control authority and harder interception.
- `eta = sigma * sqrt(dt) / L`: one-step diffusive displacement relative to
  the arena.

The normalized one-step drift and control displacements are

```text
drift / L   = rho * eta
control / L = rho * eta / kappa
```

Therefore `(rho, kappa)` do not determine the normalized dynamics. The old
two-axis plan omitted `eta`; the revised study tests that omission rather than
assuming two-axis sufficiency.

## Executed SPS-C03 anchor

The immutable SPS-C03 summaries record:

| Quantity | Executed value |
|---|---:|
| `alpha` | 0.06 |
| `sigma` | 0.06 |
| `dt` | 0.02 |
| `v_max` | 0.12 |
| `rho` | 0.1414213562 |
| `kappa` | 0.50 |
| `eta` | 0.0084852814 |
| Mean matched gain | +1.1875 captures |
| One-sided lower bound | +0.4586551 |

Earlier FR-B3 documents incorrectly described `v_max=0.30` and `kappa=0.20`.
Those values were not used by SPS-C03 and must not be cited as its anchor.

## What is implemented here

- `src/particle_benchmark/catchability.py`: dimensionless transforms and
  physically equivalent rescaling.
- `configs/experiments/fr_b3_catchability.yaml`: frozen candidate design with
  27 factorial cells and a scale-equivalence audit.
- `analysis/run_fr_b3_catchability.py`: FR-B3-only runner with matched-stream
  validation, provenance, parallel execution, and development limits.
- `analysis/analyze_fr_b3_catchability.py`: paired contrasts and held-out
  comparison of two-axis versus three-axis predictive models.
- `analysis/verify_fr_b3_factorial.py`: fail-closed completeness, pairing,
  provenance, and artifact-integrity gate for the registered run.
- `analysis/plot_fr_b3_catchability.py`: immutable publication figures, exact
  plotted-data table, chart contracts, and hashes.
- `analysis/calibrate_fr_b3_design.py`: reproducible seed-budget and decision-
  rule calibration from the historical paired variance.
- `hpc/fr_b3/`: PBS setup, preflight, submission, run, validation, analysis,
  and plotting package; no branch-switching or merge operation.
- `src/particle_benchmark/marl/representations.py`: slot-matched raw-physical
  and dimensionless observation adapters, canonical-only frozen statistics,
  and an environment wrapper usable by existing trainers.
- `src/particle_benchmark/marl/transfer.py` and
  `analysis/evaluate_fr_b3_transfer.py`: hash-checked IPPO/CommNet bundles and
  deterministic-mean scale evaluation with a CommNet zero-message ablation.
- `configs/experiments/fr_b3_learned_transfer.yaml`: unregistered candidate
  matrix whose runner refuses to execute until the remaining blockers clear.
- `tests/test_catchability.py`, `tests/test_fr_b3_pipeline.py`, and
  `tests/test_fr_b3_transfer.py`: anchor, transform, invariance, bundle,
  fail-closed execution, validation, analysis, and rendering tests.

The protocol is registered and externally timestamped. The v2
scale-equivalence audit passed all eight frozen seed-policy comparisons; see
[`RESCALING_AUDIT.md`](RESCALING_AUDIT.md) for the failed-first audit trail and
versioned correctness amendment.

## Scope boundaries

This branch does not claim that three quantities characterize arbitrary swarm
systems. The factorial freezes horizon, normalized geometry, team size,
particle density, sensing radius, capture radius, field family, and controller
definitions. It asks whether `eta` is needed within that controlled SPS slice.

Speculative mappings to agricultural, ocean, wildfire, or search-and-rescue
systems were removed: no domain calibration currently supports those claims.

See [PROTOCOL.md](PROTOCOL.md) for the experiment contract and
[DESIGN_REVIEW.md](DESIGN_REVIEW.md) for the statistical sign-off package.
[RESCALING_AUDIT.md](RESCALING_AUDIT.md) records the completed correctness gate,
and [ROADMAP.md](ROADMAP.md) covers venue fit, feasibility gates, and remaining
work. [LEARNED_TRANSFER_PROTOCOL.md](LEARNED_TRANSFER_PROTOCOL.md) is the
unregistered follow-on learning design; [LITERATURE_POSITIONING.md](LITERATURE_POSITIONING.md)
defines the source-backed novelty boundary; and [paper/README.md](paper/README.md)
explains the results-gated manuscript skeleton.
