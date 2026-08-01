"""Train one MARL architecture for one seed — HPC-parallel entry point for SPS-WO-08.

Each PBS job calls this script with a specific --arch and --train-seed.
The script trains the policy, evaluates it on the frozen eval seeds (9001-9008),
runs the same scripted baselines on those seeds for direct comparison, then
writes a self-contained JSON result.

After all 48 jobs complete (6 archs × 8 seeds), use analysis/aggregate_marl_results.py
to merge per-seed JSONs into the full SPS-WO-08 report.

Seed firewall
-------------
Training seeds 8001-8008 are reserved for SPS-WO-08 training.
Eval seeds 9001-9008 are reserved for SPS-WO-08 evaluation.
Neither range may be used for any other experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    sys.exit(
        "ERROR: PyTorch is required.\n"
        "  conda activate sps-marl\n"
        "  or: pip install 'stochastic-particle-system[marl]'"
    )

from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.marl.networks import compute_obs_dim
from particle_benchmark.marl.trainer import (
    train_ippo,
    train_mappo,
    train_maddpg,
    train_coma,
    train_commnet,
    train_vdn,
)
from particle_benchmark.dynamics.fields import field_velocity
from particle_benchmark.policies import (
    capacity_matched_velocity_controller,
    capacity_matched_velocity_controller_v2,
    full_state_interception_oracle,
    stationary_policy,
)

# ---------------------------------------------------------------------------
# Frozen experiment constants
# ---------------------------------------------------------------------------
EXPERIMENT_ID = "SPS-WO-08-MARL-BASELINES"

# Seed firewalls
TRAIN_SEED_RANGE = range(8001, 8009)
EVAL_SEEDS: tuple[int, ...] = tuple(range(9001, 9009))

FROZEN_ALPHA = 0.06
EVALUATION_STEPS = 67  # same endpoint as SPS-C03

ALL_ARCHS = ("ippo", "mappo", "commnet", "coma", "vdn", "maddpg")

TRAIN_FUNCS = {
    "ippo": train_ippo,
    "mappo": train_mappo,
    "commnet": train_commnet,
    "coma": train_coma,
    "vdn": train_vdn,
    "maddpg": train_maddpg,
}

# On-policy algorithms use n_episodes; MADDPG uses n_steps.
ONPOLICY_ARCHS = frozenset({"ippo", "mappo", "commnet", "coma", "vdn"})


# ---------------------------------------------------------------------------
# Scripted baseline evaluation (same as WO-07C for apples-to-apples comparison)
# ---------------------------------------------------------------------------

def _evaluate_scripted(
    policy_name: str,
    env: ParticleCollectorEnv,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    yields = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            if policy_name == "stationary":
                actions = stationary_policy(env.config.collector_count)
            elif policy_name == "capacity_matched_independent":
                actions = capacity_matched_velocity_controller(observations, shared=False)
            elif policy_name == "shared_summary_v2":
                actions = capacity_matched_velocity_controller_v2(observations, shared=True)
            elif policy_name == "full_state_interception_oracle":
                assert env.collector_positions is not None
                assert env.particle_positions is not None
                assert env._episode_field_kwargs is not None
                assert env.capture_state is not None
                true_motion = field_velocity(
                    env.particle_positions,
                    env.config.field_family,
                    env.config.signal_strength,
                    **env._episode_field_kwargs,
                )
                actions = full_state_interception_oracle(
                    env.collector_positions,
                    env.particle_positions,
                    true_motion,
                    env.capture_state.owner < 0,
                    collector_max_speed=env.config.collector_max_speed,
                    receding_horizon=2.0,
                )
            else:
                raise ValueError(f"Unknown scripted policy: {policy_name}")
            observations, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        yields.append(float(info["captured_total"] / env.config.particle_count))

    return {
        "policy": policy_name,
        "eval_seeds": list(seeds),
        "yields": yields,
        "mean_yield": float(np.mean(yields)),
        "std_yield": float(np.std(yields)),
        "min_yield": float(np.min(yields)),
        "max_yield": float(np.max(yields)),
    }


# ---------------------------------------------------------------------------
# Learned policy evaluation
# ---------------------------------------------------------------------------

def _evaluate_learned_onpolicy(
    policy: Any,
    env: ParticleCollectorEnv,
    seeds: tuple[int, ...],
    algorithm: str,
    train_seed: int,
) -> dict[str, Any]:
    yields = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            actions, _, _, _ = policy._get_actions_with_raw(observations)
            observations, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        yields.append(float(info["captured_total"] / env.config.particle_count))

    return {
        "algorithm": algorithm,
        "train_seed": train_seed,
        "eval_seeds": list(seeds),
        "yields": yields,
        "mean_yield": float(np.mean(yields)),
        "std_yield": float(np.std(yields)),
        "min_yield": float(np.min(yields)),
        "max_yield": float(np.max(yields)),
    }


def _evaluate_maddpg(
    policy: Any,
    env: ParticleCollectorEnv,
    seeds: tuple[int, ...],
    train_seed: int,
) -> dict[str, Any]:
    yields = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            actions = policy.get_actions(observations, explore=False)
            observations, _, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        yields.append(float(info["captured_total"] / env.config.particle_count))

    return {
        "algorithm": "maddpg",
        "train_seed": train_seed,
        "eval_seeds": list(seeds),
        "yields": yields,
        "mean_yield": float(np.mean(yields)),
        "std_yield": float(np.std(yields)),
        "min_yield": float(np.min(yields)),
        "max_yield": float(np.max(yields)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{EXPERIMENT_ID} — single (arch, seed) training run"
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=list(ALL_ARCHS),
        help="MARL architecture to train.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        required=True,
        help=f"Training seed (must be in range {TRAIN_SEED_RANGE.start}-{TRAIN_SEED_RANGE.stop - 1}).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20_000,
        help="Training episodes for on-policy algorithms (IPPO, MAPPO, CommNet, COMA, VDN).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500_000,
        help="Training environment steps for off-policy MADDPG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the per-seed JSON result.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for training checkpoints. Default: adjacent to --output.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--repository-base-commit",
        default="unknown",
        help="Git commit SHA for provenance.",
    )
    args = parser.parse_args()

    if args.train_seed not in TRAIN_SEED_RANGE:
        parser.error(
            f"--train-seed {args.train_seed} is outside the reserved SPS-WO-08 "
            f"training range {TRAIN_SEED_RANGE.start}-{TRAIN_SEED_RANGE.stop - 1}."
        )

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = args.output.parent / "checkpoints" / f"{args.arch}_seed{args.train_seed}"

    env_config = ParticleEnvConfig(
        horizon=EVALUATION_STEPS,
        signal_strength=FROZEN_ALPHA,
    )
    env = ParticleCollectorEnv(env_config)

    print(f"\n{'='*60}")
    print(f"{EXPERIMENT_ID}")
    print(f"  arch={args.arch}  train_seed={args.train_seed}")
    print(f"  n_episodes={args.n_episodes}  n_steps={args.n_steps}")
    print(f"  output={args.output}")
    print(f"  commit={args.repository_base_commit}")
    print(f"  started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # ------------------------------------------------------------------
    # Scripted baselines (same eval seeds)
    # ------------------------------------------------------------------
    print("Evaluating scripted baselines ...")
    scripted = {}
    for pol_name in (
        "stationary",
        "capacity_matched_independent",
        "shared_summary_v2",
        "full_state_interception_oracle",
    ):
        scripted[pol_name] = _evaluate_scripted(pol_name, env, EVAL_SEEDS)
        print(
            f"  {pol_name:45s}: mean_yield={scripted[pol_name]['mean_yield']:.4f}"
        )

    # ------------------------------------------------------------------
    # MARL training
    # ------------------------------------------------------------------
    print(f"\nTraining {args.arch} seed={args.train_seed} ...")
    train_fn = TRAIN_FUNCS[args.arch]

    if args.arch in ONPOLICY_ARCHS:
        history = train_fn(
            env_config,
            n_episodes=args.n_episodes,
            seed=args.train_seed,
            verbose=True,
            checkpoint_every=max(1000, args.n_episodes // 10),
            checkpoint_dir=checkpoint_dir,
            resume_from=args.resume_from,
        )
        policy = history["_trained_policy"]
        learned_eval = _evaluate_learned_onpolicy(
            policy, env, EVAL_SEEDS, args.arch, args.train_seed
        )
    else:
        # MADDPG
        history = train_fn(
            env_config,
            n_steps=args.n_steps,
            seed=args.train_seed,
            verbose=True,
            checkpoint_every=max(10_000, args.n_steps // 10),
            checkpoint_dir=checkpoint_dir,
            resume_from=args.resume_from,
        )
        policy = history["_trained_policy"]
        learned_eval = _evaluate_maddpg(policy, env, EVAL_SEEDS, args.train_seed)

    wall_time_s = time.time() - t0

    print(f"\nEval result (arch={args.arch}, train_seed={args.train_seed}):")
    print(f"  mean_yield = {learned_eval['mean_yield']:.4f} ± {learned_eval['std_yield']:.4f}")
    print(f"  vs shared_summary_v2: {scripted['shared_summary_v2']['mean_yield']:.4f}")
    print(f"  vs independent:       {scripted['capacity_matched_independent']['mean_yield']:.4f}")
    print(f"  vs oracle:            {scripted['full_state_interception_oracle']['mean_yield']:.4f}")
    print(f"  wall time: {wall_time_s/3600:.2f}h")

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "arch": args.arch,
        "train_seed": args.train_seed,
        "eval_seeds": list(EVAL_SEEDS),
        "frozen_alpha": FROZEN_ALPHA,
        "evaluation_steps": EVALUATION_STEPS,
        "n_training_episodes": args.n_episodes if args.arch in ONPOLICY_ARCHS else None,
        "n_training_steps": args.n_steps if args.arch == "maddpg" else None,
        "repository_base_commit": args.repository_base_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": wall_time_s,
        "scripted_baselines": scripted,
        "learned_eval": learned_eval,
        "training_history_summary": {
            "n_updates": len(history.get("losses", [])),
            "final_eval": history.get("final_eval"),
            "last_episode_yield": (
                history["episode_yields"][-1] if history.get("episode_yields") else None
            ),
            "resumed_from": history.get("resumed_from"),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
