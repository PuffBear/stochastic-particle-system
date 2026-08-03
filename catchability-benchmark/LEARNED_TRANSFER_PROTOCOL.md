# FR-B3 Learned Representation-Transfer Protocol

**Status:** candidate design; not registered; no transfer-training seed has been
run.  
**Purpose:** supply the learning result required for a credible ICLR submission.

## Research question

> When a multi-agent policy is trained at one physical scale, does expressing
> its observations in dimensionless variables improve zero-shot transfer to
> physically rescaled but dynamically equivalent environments?

This is deliberately distinct from the scripted rescaling audit. The audit
checks that known controllers and simulator dynamics can be made exactly scale
equivalent. This experiment asks whether a learned representation discovers or
inherits that invariance.

## Candidate claims

- **L1 — representation transfer:** dimensionless observations reduce the
  zero-shot performance drop under equivalent physical rescaling relative to
  raw physical observations.
- **L2 — architecture interaction:** the effect is estimated separately for an
  independent learner and a communicating learner; communication is not assumed
  to help transfer.
- **L3 — in-scale fidelity:** any transfer advantage must not be explained by a
  materially weaker raw or dimensionless policy at the canonical training
  scale.

Failure to support L1 is a valid negative result. Failure of L3 blocks a clean
representation claim because training quality would be confounded with transfer.

## Frozen candidate matrix

| Factor | Levels |
|---|---|
| Architecture | IPPO; CommNet |
| Observation representation | raw physical; dimensionless |
| Training scale | canonical only |
| Zero-shot evaluation scale | canonical; length x2; time x4; mixed half |
| Training seeds | candidate `8301-8305` |
| Evaluation seeds | candidate `8601-8664`, common to every trained policy |
| Tuning seeds | candidate `8701-8716`, never used in final evaluation |

IPPO is the primary non-communicating architecture because its actors neither
share parameters nor exchange messages at execution. CommNet is the primary
communicating architecture because the repository already provides its
mean-pooled continuous message channel and a zero-message execution ablation.
MAPPO is a robustness extension, not part of the minimum ICLR gate.

The candidate design contains 20 training runs (`2 x 2 x 5`) and 5,120 primary
evaluation episodes (`20 x 4 x 64`). Training and evaluation budgets must be
calibrated on the tuning panel before the status changes to registered.
The CommNet zero-message control adds 2,560 secondary evaluation episodes, for
7,680 evaluations after all 20 frozen checkpoints exist.

## Representation contract

Both representations expose the same entities, masks, ordering, history, and
number of floating-point slots. They differ only in units.

### Raw physical representation

- collector position in physical length units;
- visible-particle displacement in physical length units;
- finite-difference particle velocity in physical length/time units;
- radial distance in physical length units;
- teammate displacement in physical length units;
- binary presence and velocity-validity masks.

### Dimensionless representation

Let `L` be the arena characteristic length, `r_s` the sensing radius, and `dt`
the physical time step.

- collector position: `x / L`;
- visible-particle displacement: `(x_particle - x_collector) / r_s`;
- apparent particle velocity: `v * dt / r_s`;
- radial distance: `distance / r_s`;
- teammate displacement: `(x_teammate - x_collector) / L`;
- the same binary masks.

These features are invariant under the registered transformations
`L -> length_scale * L` and `t -> time_scale * t`. Actions remain normalized
directions and the environment applies the appropriately scaled speed limit.

For both representations, input standardization statistics are fitted only on
canonical training trajectories and then frozen. Target-scale observations
must never update running statistics. This prevents test-time adaptation from
silently solving the raw-representation condition.

## Training and evaluation rules

1. Use identical network widths, PPO settings, rollout budgets, reward, and
   stopping rules across representations within each architecture.
2. Tune only on the canonical tuning panel. Select one configuration per
   architecture before final training seeds are run.
3. Save checkpoints, learning curves, optimizer state, source commit, config
   hash, package versions, and wall-clock time for every run.
4. Evaluate frozen policies with deterministic action means as the primary
   result. A common-random-number stochastic-policy evaluation is secondary.
5. Perform no target-scale fine-tuning, normalization update, checkpoint
   selection, or early stopping.
6. For CommNet, repeat evaluation with received communication vectors zeroed.
   This is an attribution control, not a third architecture.
7. Report every training seed. A failed or collapsed run remains in the primary
   analysis unless a pre-registered infrastructure failure rule applies.

## Estimands

For architecture `a`, representation `r`, training seed `q`, evaluation seed
`s`, and rescaling `v`, let `Y[a,r,q,s,v]` be unique team capture yield.

The within-policy zero-shot change is

```text
Delta[a,r,q,v] = mean_s(Y[a,r,q,s,v] - Y[a,r,q,s,canonical]).
```

For each non-canonical scale, the representation effect is

```text
Psi[a,v] = mean_q(Delta[a,dimensionless,q,v]
                  - Delta[a,raw,q,v]).
```

Positive `Psi` favors the dimensionless representation. The primary summary is
the mean of `Psi[a,v]` over the three non-canonical scales, reported separately
for IPPO and CommNet. Also report the worst-scale `Delta`, canonical yield, and
all seed-level values. Avoid ratios because a weak canonical policy can make
retention ratios unstable.

Uncertainty uses a hierarchical paired bootstrap: resample training seeds, then
common evaluation seeds within each sampled training seed. The practical-effect
threshold and training budget remain **TBD before registration** and must be
calibrated using only tuning seeds.

## Required figures and tables

- learning curves for all 20 runs, faceted by architecture and representation;
- canonical versus each rescaled yield with paired training-seed intervals;
- representation difference-in-differences `Psi[a,v]`;
- CommNet full-message versus zero-message transfer;
- exact hyperparameter, runtime, and seed table;
- failed-run and checkpoint-completeness audit.

## Go/no-go gate for ICLR

Proceed with the ICLR framing only if:

1. the scripted factorial yields a decisive, reproducible statement about the
   minimum catchability parameterization; and
2. at least one architecture has valid in-scale training and an interpretable
   raw-versus-dimensionless zero-shot comparison.

If learned policies fail in-scale or the transfer matrix cannot be completed by
31 August 2026, retain the scripted benchmark result and target AAMAS rather
than overstating an ICLR representation contribution.

## Registration blockers

- [x] implement raw and dimensionless adapters with equality-under-rescaling,
  mask-preservation, and slot-parity tests;
- [x] add hash-checked deterministic checkpoint evaluation for IPPO and
  CommNet, including the zero-message execution control;
- run tuning-panel budget and stability pilots;
- calibrate the practical-effect threshold and final number of training seeds;
- [x] confirm that candidate seed ranges do not overlap any existing experiment
  as of branch commit used for registration;
- obtain human sign-off and an external timestamp.

## Implemented execution guard

The machine-readable candidate design is
`configs/experiments/fr_b3_learned_transfer.yaml`. Inspect it without PyTorch or
episode execution using:

```bash
PYTHONPATH=src python analysis/evaluate_fr_b3_transfer.py \
  --config configs/experiments/fr_b3_learned_transfer.yaml --dry-run
```

The dry-run derives 20 required checkpoint bundles, 5,120 primary episodes,
2,560 CommNet ablations, and 7,680 total evaluations. Outside `--dry-run`, the
runner exits before loading a checkpoint or resetting an environment while the
protocol status remains `candidate_not_registered`.

Each future checkpoint bundle must contain a model, canonical-training frozen
standardizer, and metadata recording the adapter contract, representation,
architecture, training seed, checkpoint episode, canonical-environment hash,
and artifact hashes. Evaluation uses clipped deterministic action means; it
does not sample from the PPO action distribution or update normalization.
