# Trajectory dataset contract

Compact exploratory calibration summaries now exist under `results/raw/SPS-P01`
and `results/raw/SPS-P02`. They are not a confirmatory scientific dataset and
cannot update SPS-C01. The runner can also generate schema-validated trajectory
prefixes and manifests for each pair; the compressed pilots retain only the
seed-level summaries needed for audit and analysis.

## Storage layout

The validated runner writes one immutable bundle per matched scenario:

```text
<output_dir>/
  <run_id>.null.trajectory.jsonl
  <run_id>.signal.trajectory.jsonl
  <run_id>.summary.json
  <run_id>.manifest.json
```

Trajectory rows contain positions, actions, ownership, contact events, tie
provenance, contact-model identity, and termination flags. The manifest records
the repository revision, configuration hash, scenario seed, complete matching
contract, policy identity, runtime, byte count, and SHA-256 digest of every
artifact. Writes use exclusive creation and cannot overwrite prior evidence.

## Sampling unit

The scenario seed is the independent unit. Particles, collectors, time steps,
and the two members of a matched pair must never be treated as independent
replicates.

## Privacy and leakage boundary

The released state trajectory may contain privileged simulator state for
reproducibility. Policy observations must be stored separately when included
and must pass the no-field/no-global/no-future-noise leakage tests. A policy may
not read the manifest or privileged arrays during an episode.

## Current gate

SPS-P01 and SPS-P02 are registered calibration runs with raw compact summary
paths and analysis scripts in `paper/experiments.jsonl`. No confirmatory
trajectory dataset has been authorized. A future release requires independent
seeds, a valid multi-agent mechanism, a meaningful power design, timestep
convergence, and a one-command reproduction path.
