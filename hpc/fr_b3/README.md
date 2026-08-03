# FR-B3 HPC execution package

This directory runs the registered 6,912-episode factorial and its complete
post-processing pipeline. It does not merge or update any branch.

## One-time setup

Clone only `fr-b3-catchability-benchmark`, or point `SPS_REPO_DIR` to an
existing clean checkout of that branch. The setup script deliberately refuses
to switch branches, pull, reset, or overwrite outputs.

```bash
export SPS_REPO_DIR="$HOME/sps/stochastic-particle-system"
bash hpc/fr_b3/setup.sh
```

## Submission

Inspect the exact queue command first:

```bash
bash hpc/fr_b3/submit.sh --dry-run
```

Then submit from the HPC login node:

```bash
bash hpc/fr_b3/submit.sh
```

The PBS request is CPU-only: 8 CPUs, 16 GB RAM, and six hours. Adjust only the
site-specific queue/resource header if the cluster rejects its syntax; do not
change the registered experiment command.

## Fail-closed behavior

Before any frozen seed runs, `preflight.py` requires:

- the exact `fr-b3-catchability-benchmark` branch;
- a clean tracked worktree and a full commit SHA;
- registered protocol status;
- absent immutable run, validation, analysis, and figure output paths;
- a dry-run containing exactly 27 cells, 64 seeds, four policies, and 6,912
  episodes.

After execution, the job verifies the exact condition-seed-policy Cartesian
product, paired streams, value domains, artifact hashes, and source blobs at the
manifest commit. Analysis and figures run only after this gate passes.

## Expected outputs

```text
results/raw/FR-B3-CATCHABILITY-FACTORIAL/
results/analysis/FR-B3-CATCHABILITY-FACTORIAL-VALIDATION.json
results/analysis/FR-B3-CATCHABILITY-FACTORIAL.json
results/figures/FR-B3-CATCHABILITY-FACTORIAL/
```

Do not rerun into an existing path. Preserve any failure output and diagnose it
under a new versioned path, following the rescaling-audit precedent.
