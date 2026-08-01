#!/usr/bin/env bash
# One-time ShARC HPC setup for the stochastic-particle-system project.
# Run this from the login node once before submitting any jobs.
#
# Usage:
#   bash hpc/setup.sh
set -euo pipefail

SPS_DIR="$HOME/sps"
REPO_DIR="$SPS_DIR/stochastic-particle-system"
RESULTS_DIR="$REPO_DIR/results/raw/SPS-WO-08-MARL"
LOGS_DIR="$REPO_DIR/hpc/logs"

echo "=== SPS HPC setup ==="
echo "Target directory: $REPO_DIR"

# 1. Create project directory
mkdir -p "$SPS_DIR"

# 2. Clone repo (skip if already present)
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already cloned at $REPO_DIR — pulling latest."
    git -C "$REPO_DIR" pull
else
    echo "Cloning repo..."
    git clone https://github.com/puffbear/stochastic-particle-system.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# Make sure we're on the right branch
git checkout claude/research-autonomy-review-rwjyjg

# 3. Create results and log directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# 4. Create conda environment
module load Anaconda3

if conda env list | grep -q "^sps-marl "; then
    echo "Conda env sps-marl already exists — updating."
    conda env update -f hpc/environment.yml --prune
else
    echo "Creating conda env sps-marl..."
    conda env create -f hpc/environment.yml
fi

# 5. Smoke test
echo ""
echo "=== Smoke test ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sps-marl

python - <<'PYEOF'
import torch
print(f"PyTorch {torch.__version__} — GPU: {torch.cuda.is_available()}")
from particle_benchmark.environment import ParticleEnvConfig, ParticleCollectorEnv
from particle_benchmark.marl.trainer import train_ippo
config = ParticleEnvConfig(horizon=67, signal_strength=0.06)
h = train_ippo(config, n_episodes=20, eval_every=1000, seed=99, rollout_steps=400, verbose=False)
print(f"IPPO smoke test passed — final yield: {h['final_eval']['mean_yield']:.3f}")
PYEOF

echo ""
echo "=== Setup complete ==="
echo "Submit jobs with: bash hpc/submit_all.sh"
