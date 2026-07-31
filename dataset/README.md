# Trajectory dataset contract

No scientific dataset has been generated yet. These files freeze the shape of
future artifacts so a smoke run cannot silently become an undocumented result.

## Storage layout

Each matched scenario produces one immutable run directory:

```text
results/raw/<experiment_id>/<run_id>/
  manifest.json
  null.npz
  signal.npz
  events.jsonl
```

The two NPZ files contain dense arrays for particle positions, collector
positions, collector actions, and ownership. `events.jsonl` contains sparse
capture events. `manifest.json` records the repository revision, complete
configuration hash, scenario seed, matching contract, policy identity, runtime,
and SHA-256 digest of every artifact.

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

The schemas are design artifacts only. A dataset is not considered generated
until registered runs name raw paths, checksums, validation commands, and an
analysis script in `paper/experiments.jsonl`.
