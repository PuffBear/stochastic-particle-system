#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="${SPS_REPO_DIR:-$HOME/sps/stochastic-particle-system}"
EXPECTED_BRANCH="fr-b3-catchability-benchmark"

if [[ ! -d "$REPOSITORY_ROOT/.git" ]]; then
  echo "Repository not found at $REPOSITORY_ROOT" >&2
  echo "Clone only the FR-B3 branch before setup:" >&2
  echo "git clone --single-branch --branch $EXPECTED_BRANCH https://github.com/PuffBear/stochastic-particle-system.git $REPOSITORY_ROOT" >&2
  exit 1
fi

cd "$REPOSITORY_ROOT"
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "Wrong branch: $CURRENT_BRANCH (expected $EXPECTED_BRANCH)" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked worktree changes are present; setup refuses to alter them." >&2
  exit 1
fi

module load Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx 'sps-fr-b3'; then
  conda env update -n sps-fr-b3 -f hpc/fr_b3/environment.yml --prune
else
  conda env create -f hpc/fr_b3/environment.yml
fi
conda activate sps-fr-b3
python -m pip install -e '.[test,fr-b3-report]'

export PYTHONPATH="$REPOSITORY_ROOT/src"
python -m pytest -q
python hpc/fr_b3/preflight.py \
  --repository-root "$REPOSITORY_ROOT" \
  --config configs/experiments/fr_b3_catchability.yaml \
  --run-output results/raw/FR-B3-CATCHABILITY-FACTORIAL \
  --validation-output results/analysis/FR-B3-CATCHABILITY-FACTORIAL-VALIDATION.json \
  --analysis-output results/analysis/FR-B3-CATCHABILITY-FACTORIAL.json \
  --figure-output results/figures/FR-B3-CATCHABILITY-FACTORIAL

echo "FR-B3 HPC environment is ready at commit $(git rev-parse HEAD)."
