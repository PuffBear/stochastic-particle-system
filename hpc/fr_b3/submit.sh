#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash hpc/fr_b3/submit.sh [--dry-run]" >&2
  exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"
mkdir -p hpc/logs

COMMAND=(
  qsub
  -o hpc/logs/frb3_factorial.out
  -e hpc/logs/frb3_factorial.err
  -v "SPS_REPO_DIR=$REPOSITORY_ROOT"
  hpc/fr_b3/job.pbs
)

if [[ $DRY_RUN -eq 1 ]]; then
  printf 'dry-run:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

command -v qsub >/dev/null 2>&1 || {
  echo "qsub is unavailable; run this helper on the HPC login node." >&2
  exit 1
}
"${COMMAND[@]}"
