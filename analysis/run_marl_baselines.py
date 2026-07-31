"""Train IPPO and MAPPO baselines and compare against scripted baselines.

Experiment: SPS-WO-08-MARL-BASELINES
======================================
Trains IPPO and MAPPO across multiple seeds, evaluates on diagnostic eval
seeds, and compares against stationary / local_flow_v1 /
capacity_matched_independent scripted baselines on the same eval seeds.

Usage
-----
    python analysis/run_marl_baselines.py \\
        --output results/sps-wo-08-marl-baselines.json \\
        --repository-base-commit <SHA>

If --dry-run is passed, only 2 training seeds and 10 episodes are used so
the script completes quickly for debugging.

NOTE: Requires PyTorch.  If torch is not installed the script exits with a
clear error message.  Install with:
    pip install 'stochastic-particle-system[marl]'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Early import check for torch
# ---------------------------------------------------------------------------
try:
    import torch  # noqa: F401
except ImportError:
    sys.exit(
        "ERROR: PyTorch is required for MARL baselines.\n"
        "Install with: pip install 'stochastic-particle-system[marl]'"
    )

from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.marl.ippo import IPPO
from particle_benchmark.marl.mappo import MAPPO
from particle_benchmark.marl.maddpg import MADDPG
from particle_benchmark.marl.coma import COMA
from particle_benchmark.marl.commnet import CommNet
from particle_benchmark.marl.vdn import VDN
from particle_benchmark.marl.networks import compute_obs_dim
from particle_benchmark.marl.trainer import (
    train_ippo,
    train_mappo,
    train_maddpg,
    train_coma,
    train_commnet,
    train_vdn,
)
from particle_benchmark.policies import (
    capacity_matched_velocity_controller,
    local_flow_v1_policy,
    stationary_policy,
)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
EXPERIMENT_ID = "SPS-WO-08-MARL-BASELINES"
TRAIN_SEEDS = tuple(range(5001, 5017))   # 16 training seeds
EVAL_SEEDS = tuple(range(5501, 5509))    # 8 eval seeds (diagnostic only, ineligible for confirmation)
FROZEN_ALPHA = 0.06
EVALUATION_STEPS = 67
N_TRAINING_EPISODES = 500
ALL_ALGORITHMS = ("ippo", "mappo", "maddpg", "coma", "commnet", "vdn")


# ---------------------------------------------------------------------------
# Scripted baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_scripted_baseline(
    policy_name: str,
    env: ParticleCollectorEnv,
    seeds: tuple[int, ...],
) -> dict:
    """Evaluate one scripted baseline policy over multiple seeds."""
    yields = []
    for seed in seeds:
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            if policy_name == "stationary":
                actions = stationary_policy(env.config.collector_count)
            elif policy_name == "local_flow_v1":
                actions = local_flow_v1_policy(observations)
            elif policy_name == "capacity_matched_independent":
                actions = capacity_matched_velocity_controller(observations, shared=False)
            else:
                raise ValueError(f"Unknown scripted baseline: {policy_name}")
            observations, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        yield_frac = info["captured_total"] / env.config.particle_count
        yields.append(float(yield_frac))

    return {
        "policy": policy_name,
        "seeds": list(seeds),
        "yields": yields,
        "mean_yield": float(np.mean(yields)),
        "std_yield": float(np.std(yields)),
        "min_yield": float(np.min(yields)),
        "max_yield": float(np.max(yields)),
    }


def evaluate_learned_policy(
    policy: IPPO | MAPPO | COMA | CommNet | VDN,
    env: ParticleCollectorEnv,
    seeds: tuple[int, ...],
    algorithm: str,
    train_seed: int,
) -> dict:
    """Evaluate a trained on-policy algorithm over diagnostic eval seeds."""
    yields = []
    for seed in seeds:
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            actions, _, _, _ = policy._get_actions_with_raw(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        yield_frac = info["captured_total"] / env.config.particle_count
        yields.append(float(yield_frac))

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


def evaluate_maddpg_policy(
    policy: MADDPG,
    env: ParticleCollectorEnv,
    seeds: tuple[int, ...],
    train_seed: int,
) -> dict:
    """Evaluate a trained MADDPG policy in deterministic mode."""
    yields = []
    for seed in seeds:
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            actions = policy.get_actions(observations, explore=False)
            observations, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        yield_frac = info["captured_total"] / env.config.particle_count
        yields.append(float(yield_frac))

    return {
        "algorithm": "MADDPG",
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
    parser = argparse.ArgumentParser(description=f"Run {EXPERIMENT_ID}")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "sps-wo-08-marl-baselines.json",
        help="Path to write the JSON results report.",
    )
    parser.add_argument(
        "--repository-base-commit",
        default="unknown",
        help="Git commit SHA for provenance tracking.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use minimal seeds/episodes for quick debugging.",
    )
    parser.add_argument(
        "--n-training-episodes",
        type=int,
        default=N_TRAINING_EPISODES,
        help="Number of training episodes per seed (on-policy algorithms).",
    )
    parser.add_argument(
        "--n-training-steps",
        type=int,
        default=100_000,
        help="Number of environment steps per seed (MADDPG off-policy).",
    )
    parser.add_argument(
        "--skip-mappo",
        action="store_true",
        help="Skip MAPPO training (legacy flag; use --algorithms instead).",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(ALL_ALGORITHMS),
        default=list(ALL_ALGORITHMS),
        metavar="ALG",
        help=(
            f"Algorithms to train (default: all). "
            f"Choices: {', '.join(ALL_ALGORITHMS)}"
        ),
    )
    args = parser.parse_args()

    train_seeds = TRAIN_SEEDS[:2] if args.dry_run else TRAIN_SEEDS
    eval_seeds = EVAL_SEEDS[:2] if args.dry_run else EVAL_SEEDS
    n_episodes = 10 if args.dry_run else args.n_training_episodes
    n_steps = 2000 if args.dry_run else args.n_training_steps

    # Respect legacy --skip-mappo flag.
    algorithms = [a for a in args.algorithms if not (a == "mappo" and args.skip_mappo)]

    env_config = ParticleEnvConfig(horizon=EVALUATION_STEPS, signal_strength=FROZEN_ALPHA)
    env = ParticleCollectorEnv(env_config)

    report: dict = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_base_commit": args.repository_base_commit,
        "env_config": {
            "horizon": env_config.horizon,
            "signal_strength": env_config.signal_strength,
            "particle_count": env_config.particle_count,
            "collector_count": env_config.collector_count,
        },
        "train_seeds": list(train_seeds),
        "eval_seeds": list(eval_seeds),
        "algorithms_run": algorithms,
        "n_training_episodes": n_episodes,
        "n_training_steps_maddpg": n_steps,
        "note": (
            "Eval seeds are diagnostic only and ineligible for confirmation. "
            "MARL training uses per-seed random initialisation; results may "
            "vary without fixing torch global seed."
        ),
        "scripted_baselines": {},
        "ippo_results": [],
        "mappo_results": [],
        "maddpg_results": [],
        "coma_results": [],
        "commnet_results": [],
        "vdn_results": [],
    }

    # ------------------------------------------------------------------
    # Scripted baselines
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluating scripted baselines ...")
    print(f"{'='*60}")
    for policy_name in ("stationary", "local_flow_v1", "capacity_matched_independent"):
        t0 = time.time()
        result = evaluate_scripted_baseline(policy_name, env, eval_seeds)
        elapsed = time.time() - t0
        result["wall_time_s"] = elapsed
        report["scripted_baselines"][policy_name] = result
        print(
            f"  {policy_name:45s}: "
            f"mean_yield={result['mean_yield']:.3f} "
            f"±{result['std_yield']:.3f}  ({elapsed:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Derive obs_dim once
    # ------------------------------------------------------------------
    init_obs, _ = env.reset(seed=train_seeds[0])
    obs_dim = compute_obs_dim(init_obs)
    print(f"\nFlat obs_dim = {obs_dim}")

    # ------------------------------------------------------------------
    # Helper: build a per-seed result dict from a history
    # ------------------------------------------------------------------
    def _make_result(history: dict, train_seed: int, n_ep: int) -> dict:
        final_eval = history.get("final_eval", {})
        return {
            "train_seed": train_seed,
            "n_training_episodes": n_ep,
            "training_history_summary": {
                "n_updates": len(history.get("losses", [])),
                "final_loss": history["losses"][-1] if history.get("losses") else None,
            },
            "final_eval_fixed_seeds": final_eval,
        }

    # ------------------------------------------------------------------
    # IPPO training
    # ------------------------------------------------------------------
    if "ippo" in algorithms:
        print(f"\n{'='*60}")
        print(f"Training IPPO across {len(train_seeds)} seeds ...")
        print(f"{'='*60}")
        for train_seed in train_seeds:
            print(f"\n[IPPO] seed={train_seed} ...")
            t0 = time.time()
            history = train_ippo(
                env_config,
                n_episodes=n_episodes,
                eval_every=max(1, n_episodes // 5),
                seed=train_seed,
                verbose=False,
            )
            elapsed = time.time() - t0
            final_eval = history.get("final_eval", {})
            result = _make_result(history, train_seed, n_episodes)
            result["wall_time_s"] = elapsed
            report["ippo_results"].append(result)
            print(
                f"  [IPPO] seed={train_seed}: "
                f"final_eval_yield={final_eval.get('mean_yield', float('nan')):.3f}  "
                f"({elapsed:.1f}s)"
            )

    # ------------------------------------------------------------------
    # MAPPO training
    # ------------------------------------------------------------------
    if "mappo" in algorithms:
        print(f"\n{'='*60}")
        print(f"Training MAPPO across {len(train_seeds)} seeds ...")
        print(f"{'='*60}")
        for train_seed in train_seeds:
            print(f"\n[MAPPO] seed={train_seed} ...")
            t0 = time.time()
            history = train_mappo(
                env_config,
                n_episodes=n_episodes,
                eval_every=max(1, n_episodes // 5),
                seed=train_seed,
                verbose=False,
            )
            elapsed = time.time() - t0
            final_eval = history.get("final_eval", {})
            result = _make_result(history, train_seed, n_episodes)
            result["wall_time_s"] = elapsed
            report["mappo_results"].append(result)
            print(
                f"  [MAPPO] seed={train_seed}: "
                f"final_eval_yield={final_eval.get('mean_yield', float('nan')):.3f}  "
                f"({elapsed:.1f}s)"
            )

    # ------------------------------------------------------------------
    # MADDPG training  (off-policy)
    # ------------------------------------------------------------------
    if "maddpg" in algorithms:
        print(f"\n{'='*60}")
        print(f"Training MADDPG across {len(train_seeds)} seeds ({n_steps} steps each) ...")
        print(f"{'='*60}")
        for train_seed in train_seeds:
            print(f"\n[MADDPG] seed={train_seed} ...")
            t0 = time.time()
            history = train_maddpg(
                env_config,
                n_steps=n_steps,
                eval_every=max(1, n_steps // 5),
                seed=train_seed,
                verbose=False,
            )
            elapsed = time.time() - t0
            final_eval = history.get("final_eval", {})
            result = {
                "train_seed": train_seed,
                "n_training_steps": n_steps,
                "training_history_summary": {
                    "n_updates": len(history.get("losses", [])),
                    "final_loss": history["losses"][-1] if history.get("losses") else None,
                },
                "final_eval_fixed_seeds": final_eval,
                "wall_time_s": elapsed,
            }
            report["maddpg_results"].append(result)
            print(
                f"  [MADDPG] seed={train_seed}: "
                f"final_eval_yield={final_eval.get('mean_yield', float('nan')):.3f}  "
                f"({elapsed:.1f}s)"
            )

    # ------------------------------------------------------------------
    # COMA training
    # ------------------------------------------------------------------
    if "coma" in algorithms:
        print(f"\n{'='*60}")
        print(f"Training COMA across {len(train_seeds)} seeds ...")
        print(f"{'='*60}")
        for train_seed in train_seeds:
            print(f"\n[COMA] seed={train_seed} ...")
            t0 = time.time()
            history = train_coma(
                env_config,
                n_episodes=n_episodes,
                eval_every=max(1, n_episodes // 5),
                seed=train_seed,
                verbose=False,
            )
            elapsed = time.time() - t0
            final_eval = history.get("final_eval", {})
            result = _make_result(history, train_seed, n_episodes)
            result["wall_time_s"] = elapsed
            report["coma_results"].append(result)
            print(
                f"  [COMA] seed={train_seed}: "
                f"final_eval_yield={final_eval.get('mean_yield', float('nan')):.3f}  "
                f"({elapsed:.1f}s)"
            )

    # ------------------------------------------------------------------
    # CommNet training
    # ------------------------------------------------------------------
    if "commnet" in algorithms:
        print(f"\n{'='*60}")
        print(f"Training CommNet across {len(train_seeds)} seeds ...")
        print(f"{'='*60}")
        for train_seed in train_seeds:
            print(f"\n[CommNet] seed={train_seed} ...")
            t0 = time.time()
            history = train_commnet(
                env_config,
                n_episodes=n_episodes,
                eval_every=max(1, n_episodes // 5),
                seed=train_seed,
                verbose=False,
            )
            elapsed = time.time() - t0
            final_eval = history.get("final_eval", {})
            result = _make_result(history, train_seed, n_episodes)
            result["wall_time_s"] = elapsed
            report["commnet_results"].append(result)
            print(
                f"  [CommNet] seed={train_seed}: "
                f"final_eval_yield={final_eval.get('mean_yield', float('nan')):.3f}  "
                f"({elapsed:.1f}s)"
            )

    # ------------------------------------------------------------------
    # VDN training
    # ------------------------------------------------------------------
    if "vdn" in algorithms:
        print(f"\n{'='*60}")
        print(f"Training VDN across {len(train_seeds)} seeds ...")
        print(f"{'='*60}")
        for train_seed in train_seeds:
            print(f"\n[VDN] seed={train_seed} ...")
            t0 = time.time()
            history = train_vdn(
                env_config,
                n_episodes=n_episodes,
                eval_every=max(1, n_episodes // 5),
                seed=train_seed,
                verbose=False,
            )
            elapsed = time.time() - t0
            final_eval = history.get("final_eval", {})
            result = _make_result(history, train_seed, n_episodes)
            result["wall_time_s"] = elapsed
            report["vdn_results"].append(result)
            print(
                f"  [VDN] seed={train_seed}: "
                f"final_eval_yield={final_eval.get('mean_yield', float('nan')):.3f}  "
                f"({elapsed:.1f}s)"
            )

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport written to: {args.output}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, result in report["scripted_baselines"].items():
        print(f"  scripted/{name:40s}: mean_yield={result['mean_yield']:.4f}")
    for alg_key, label in [
        ("ippo_results", "IPPO"),
        ("mappo_results", "MAPPO"),
        ("maddpg_results", "MADDPG"),
        ("coma_results", "COMA"),
        ("commnet_results", "CommNet"),
        ("vdn_results", "VDN"),
    ]:
        for r in report[alg_key]:
            ey = r["final_eval_fixed_seeds"].get("mean_yield", float("nan"))
            print(
                f"  {label} seed={r['train_seed']:5d}: "
                f"final_eval_yield={ey:.4f}"
            )


if __name__ == "__main__":
    main()
