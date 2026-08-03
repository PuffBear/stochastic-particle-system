#!/usr/bin/env bash
# Submit one PBS job per (architecture, seed) pair.
# Run from the login node:  bash hpc/submit_all.sh [--dry-run] [--arch ARCH]
#
# With 6 architectures × 8 seeds = 48 jobs.  The GPU queue has 4 V100s so
# at most 4 jobs run simultaneously; expect ~3-4 days to clear the queue.
#
# Flags:
#   --dry-run   Print qsub commands without submitting.
#   --arch ALG  Submit only this architecture (repeat for multiple).
#   --seeds S   Space-separated list of seeds (default: all 8).
set -euo pipefail

cd "$HOME/sps/stochastic-particle-system"

ALL_ARCHS=(ippo mappo commnet coma vdn maddpg)
ALL_SEEDS=(8001 8002 8003 8004 8005 8006 8007 8008)
N_EPISODES=20000
N_STEPS=500000
DRY_RUN=0
SELECTED_ARCHS=()
SELECTED_SEEDS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        --arch)    SELECTED_ARCHS+=("$2"); shift 2 ;;
        --seeds)   read -ra SELECTED_SEEDS <<< "$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

ARCHS=("${SELECTED_ARCHS[@]:-${ALL_ARCHS[@]}}")
SEEDS=("${SELECTED_SEEDS[@]:-${ALL_SEEDS[@]}}")

mkdir -p hpc/logs
mkdir -p results/raw/SPS-WO-08-MARL

SUBMITTED=0
for arch in "${ARCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        jobname="${arch}_${seed}"
        out="hpc/logs/${jobname}.out"
        err="hpc/logs/${jobname}.err"

        cmd="qsub -v ARCH=${arch},SEED=${seed},N_EPISODES=${N_EPISODES},N_STEPS=${N_STEPS} \
             -N ${jobname} \
             -o ${out} -e ${err} \
             hpc/job.pbs"

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[dry-run] $cmd"
        else
            echo -n "Submitting ${arch} seed=${seed} ... "
            eval "$cmd"
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
done

if [[ $DRY_RUN -eq 0 ]]; then
    echo ""
    echo "Submitted ${SUBMITTED} jobs. Monitor with: qstat -a"
    echo "Results will appear in: results/raw/SPS-WO-08-MARL/"
fi
