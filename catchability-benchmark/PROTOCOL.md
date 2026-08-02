# FR-B3 Candidate Protocol

**Status:** registered on 3 August 2026 through the immutable Git history of
`fr-b3-catchability-benchmark`; no FR-B3 registered seeds have been run at the
time of registration.

The author confirmed the complete design-review checklist before registration.
The registration commit freezes the axis grid, seed panels, policy arms,
decision rule, and analysis settings below. Its full commit SHA must be recorded
in every frozen-run manifest.

Development seed 7201 was used once for a local runner smoke test and is
explicitly excluded from the candidate rescaling-audit panel.

## Claims under test

### C1: two-axis predictive sufficiency

The coordination-gain surface can be predicted from `(rho, kappa)` without
knowing `eta`.

This is evaluated predictively, not by in-sample `R^2`. A quadratic response
surface using `(rho, kappa)` is compared with the corresponding surface using
`(rho, kappa, eta)` under leave-one-cell-out prediction. Common-seed bootstrap
resampling supplies uncertainty for the RMSE improvement.

The proposed rejection rule requires both:

1. a practically meaningful observed predictive improvement,

```text
RMSE(three-axis) / RMSE(two-axis) <= 0.80;
```

2. statistical support: the one-sided 95% bootstrap lower bound for
   `RMSE(two-axis) - RMSE(three-axis)` must exceed zero.

Non-rejection is not treated as proof of sufficiency.

### C2: dimensionless rescaling correctness

When `(rho, kappa, eta)`, horizon, and normalized geometry are fixed, physically
rescaled environments should have identical normalized trajectories under the
scripted controllers and common random numbers.

The audit requires identical capture yield and identical quantized normalized
final-state checksums for every seed-policy pair. This is a correctness gate,
not independent scientific evidence.

### C3: coordination-gain regime

Estimate

```text
D(rho, kappa, eta) = E[Y_shared - Y_independent]
```

for all factorial cells. Report every cell and every frozen seed regardless of
sign. Stationary and full-state oracle arms are diagnostics, not substitutes
for the primary capacity-matched contrast.

## Factorial design

The design is `3 x 3 x 3`:

| Axis | Low | Anchor | High |
|---|---:|---:|---:|
| `rho` | 0.0707107 | 0.1414214 | 0.2828427 |
| `kappa` | 0.25 | 0.50 | 1.00 |
| `eta` | 0.00424264 | 0.00848528 | 0.01697056 |

At fixed `dt=0.02` and `L=1`, each cell is translated to physical parameters
by

```text
sigma = eta * L / sqrt(dt)
alpha = rho * sigma / sqrt(dt)
v_max = alpha / kappa
```

The middle cell exactly recovers the executed SPS-C03 parameters.

## Frozen structure

| Setting | Value |
|---|---:|
| Arena | unit square |
| Horizon | 67 steps |
| Particles | 256 |
| Collectors | 4 |
| Sensing radius | 0.16 |
| Capture radius | 0.012 |
| Field | uniform, random orientation per seed |
| Factorial seeds | 7101-7164, shared across all cells |

Using the same fresh 64 seeds in every cell makes within-cell policy contrasts
and across-cell response differences paired. SPS-C03 seeds 6001-6032 remain
historical and are not reused in the new inferential analysis.

The 64-seed budget is calibrated to at least 80% simulated power for a broad
one-capture high-versus-low `eta` effect under zero cross-cell seed correlation,
using the historical SPS-C03 paired SD of 2.442. This does not power the study
for an effect isolated to one cell.

## Prohibited analysis changes after registration

- no diagnostic-seed gate followed by selective extra seeds;
- no dropping negative or unstable cells;
- no data-dependent offset before a log transform;
- no replacing the capacity-matched primary baseline after viewing results;
- no changing the `0.80` predictive-improvement threshold after viewing
  registered seeds;
- no presenting the rescaling correctness audit as an independent replication.

Any post-registration change must be versioned, justified, and labelled
exploratory.

## Execution

Dry-run the design:

```bash
PYTHONPATH=src python analysis/run_fr_b3_catchability.py \
  --config configs/experiments/fr_b3_catchability.yaml \
  --study factorial --dry-run
```

Run the scale audit, then the full factorial:

```bash
PYTHONPATH=src python analysis/run_fr_b3_catchability.py \
  --config configs/experiments/fr_b3_catchability.yaml \
  --study rescaling-audit --jobs 8 \
  --repository-commit <40-character-commit> \
  --output results/raw/FR-B3-CATCHABILITY-RESCALING

PYTHONPATH=src python analysis/run_fr_b3_catchability.py \
  --config configs/experiments/fr_b3_catchability.yaml \
  --study factorial --jobs 8 \
  --repository-commit <40-character-commit> \
  --output results/raw/FR-B3-CATCHABILITY-FACTORIAL
```

These complete commands are enabled only after the config's protocol status is
changed to `registered` following advisor approval and an external timestamp.

Analyze only complete immutable outputs. Development runs created with
`--max-cells` or `--max-seeds` are marked incomplete in the manifest and are
never confirmation-eligible.
