"""Training loops for IPPO and MAPPO on the particle-collector benchmark.

Usage
-----
    from particle_benchmark.environment import ParticleEnvConfig
    from particle_benchmark.marl.trainer import train_ippo, train_mappo

    config = ParticleEnvConfig(horizon=67, signal_strength=0.06)
    history = train_ippo(config, n_episodes=500)
    history = train_mappo(config, n_episodes=500)

Each function returns a training-history dict with:
    episode_yields : list of float  (captures per episode / particle_count)
    eval_yields    : list of dict   (eval results every eval_every episodes)
    policy_stats   : dict           (final log_std etc.)
    algorithm      : str

Requires PyTorch: pip install 'stochastic-particle-system[marl]'
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError(
        "MARL baselines require PyTorch: pip install 'stochastic-particle-system[marl]'"
    )

from ..environment import ParticleCollectorEnv, ParticleEnvConfig
from .ippo import IPPO
from .mappo import MAPPO
from .networks import compute_obs_dim, flatten_all_observations


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _eval_policy(
    policy: IPPO | MAPPO,
    env: ParticleCollectorEnv,
    eval_seeds: tuple[int, ...],
) -> dict[str, float]:
    """Evaluate policy over several seeds; return mean capture yield statistics."""
    yields = []
    for seed in eval_seeds:
        observations, _ = env.reset(seed=seed)
        total_captures = 0
        done = False
        while not done:
            if isinstance(policy, IPPO):
                actions, _, _, _ = policy._get_actions_with_raw(observations)
            else:
                actions, _, _, _ = policy._get_actions_with_raw(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            total_captures += int(info["captured_total"])
            done = terminated or truncated
        # Final captured_total
        final_captures = info["captured_total"]
        yield_fraction = final_captures / env.config.particle_count
        yields.append(yield_fraction)
    return {
        "mean_yield": float(np.mean(yields)),
        "std_yield": float(np.std(yields)),
        "min_yield": float(np.min(yields)),
        "max_yield": float(np.max(yields)),
        "n_seeds": len(eval_seeds),
    }


def _run_one_episode(
    policy: IPPO | MAPPO,
    env: ParticleCollectorEnv,
    seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    """Run one episode and return (total_captures, step_info_list)."""
    observations, _ = env.reset(seed=seed)
    done = False
    step_infos: list[dict[str, Any]] = []
    while not done:
        if isinstance(policy, IPPO):
            actions, _, _, _ = policy._get_actions_with_raw(observations)
        else:
            actions, _, _, _ = policy._get_actions_with_raw(observations)
        observations, reward, terminated, truncated, info = env.step(actions)
        step_infos.append(info)
        done = terminated or truncated
    return info["captured_total"], step_infos


# ---------------------------------------------------------------------------
# IPPO training loop
# ---------------------------------------------------------------------------

def train_ippo(
    env_config: ParticleEnvConfig,
    n_episodes: int = 500,
    eval_every: int = 50,
    seed: int = 9001,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train IPPO and return training history.

    One training iteration collects ``rollout_steps`` environment steps
    (spanning multiple episodes if needed), then runs a PPO update.
    ``n_episodes`` controls how many *full episode yields* we track, not
    how many PPO updates we perform.

    Parameters
    ----------
    env_config   : environment configuration
    n_episodes   : total training episodes to track yields for
    eval_every   : evaluate on a fixed set of seeds every N tracked episodes
    seed         : base seed; training episodes use seed, seed+1, ...
    rollout_steps: steps to collect per PPO update
    """
    env = ParticleCollectorEnv(env_config)
    # Derive obs_dim from one reset.
    init_obs, _ = env.reset(seed=seed)
    obs_dim = compute_obs_dim(init_obs)

    ippo = IPPO(
        obs_dim=obs_dim,
        n_agents=env_config.collector_count,
        lr=lr,
        gamma=gamma,
        clip_eps=clip_eps,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    eval_seeds_fixed = tuple(range(seed + 10000, seed + 10008))

    history: dict[str, Any] = {
        "algorithm": "IPPO",
        "obs_dim": obs_dim,
        "episode_yields": [],
        "eval_yields": [],
        "losses": [],
        "wall_times": [],
    }

    episode_seed = seed
    episode_count = 0
    t0 = time.time()

    while episode_count < n_episodes:
        # Collect a rollout (may span multiple episodes).
        rollout = ippo.collect_rollout(env, n_steps=rollout_steps, seed=episode_seed)
        losses = ippo.update(rollout)
        history["losses"].append(losses)

        # Count how many complete episodes were in this rollout (episodes end
        # at 'done' boundaries).
        dones_any = rollout["dones"][:, 0]  # shape (T,)
        n_done = int(np.sum(dones_any))
        episode_count += max(1, n_done)
        episode_seed += max(1, n_done)

        # Track yield from the last completed episode in the rollout.
        total_reward = float(rollout["rewards"].sum() / max(1, n_done))
        yield_frac = total_reward / env_config.particle_count
        history["episode_yields"].append(yield_frac)
        history["wall_times"].append(time.time() - t0)

        if verbose and len(history["episode_yields"]) % 10 == 0:
            print(
                f"[IPPO] approx episode {episode_count}/{n_episodes}, "
                f"yield={yield_frac:.3f}, "
                f"policy_loss={losses['policy_loss']:.4f}"
            )

        # Periodic evaluation.
        if episode_count % eval_every < max(1, n_done):
            eval_result = _eval_policy(ippo, env, eval_seeds_fixed)
            eval_result["approx_episode"] = episode_count
            history["eval_yields"].append(eval_result)
            if verbose:
                print(
                    f"[IPPO] EVAL ep~{episode_count}: "
                    f"mean_yield={eval_result['mean_yield']:.3f} "
                    f"±{eval_result['std_yield']:.3f}"
                )

    # Final evaluation.
    final_eval = _eval_policy(ippo, env, eval_seeds_fixed)
    final_eval["approx_episode"] = episode_count
    history["final_eval"] = final_eval

    # Policy stats.
    with torch.no_grad():
        history["policy_stats"] = {
            f"agent_{i}_log_std": ippo.networks[i].log_std.tolist()
            for i in range(env_config.collector_count)
        }

    if verbose:
        print(
            f"[IPPO] Training done. Final eval yield: "
            f"{final_eval['mean_yield']:.3f} ± {final_eval['std_yield']:.3f}"
        )

    return history


# ---------------------------------------------------------------------------
# MAPPO training loop
# ---------------------------------------------------------------------------

def train_mappo(
    env_config: ParticleEnvConfig,
    n_episodes: int = 500,
    eval_every: int = 50,
    seed: int = 9001,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train MAPPO and return training history.

    Architecture: shared actor (parameter sharing) + centralised critic.

    See :func:`train_ippo` for parameter documentation.
    """
    env = ParticleCollectorEnv(env_config)
    init_obs, _ = env.reset(seed=seed)
    obs_dim = compute_obs_dim(init_obs)

    mappo = MAPPO(
        obs_dim=obs_dim,
        n_agents=env_config.collector_count,
        lr=lr,
        gamma=gamma,
        clip_eps=clip_eps,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    eval_seeds_fixed = tuple(range(seed + 10000, seed + 10008))

    history: dict[str, Any] = {
        "algorithm": "MAPPO",
        "obs_dim": obs_dim,
        "episode_yields": [],
        "eval_yields": [],
        "losses": [],
        "wall_times": [],
    }

    episode_seed = seed
    episode_count = 0
    t0 = time.time()

    while episode_count < n_episodes:
        rollout = mappo.collect_rollout(env, n_steps=rollout_steps, seed=episode_seed)
        losses = mappo.update(rollout)
        history["losses"].append(losses)

        dones_any = rollout["dones"][:, 0]
        n_done = int(np.sum(dones_any))
        episode_count += max(1, n_done)
        episode_seed += max(1, n_done)

        total_reward = float(rollout["rewards"].sum() / max(1, n_done))
        yield_frac = total_reward / env_config.particle_count
        history["episode_yields"].append(yield_frac)
        history["wall_times"].append(time.time() - t0)

        if verbose and len(history["episode_yields"]) % 10 == 0:
            print(
                f"[MAPPO] approx episode {episode_count}/{n_episodes}, "
                f"yield={yield_frac:.3f}, "
                f"policy_loss={losses['policy_loss']:.4f}"
            )

        if episode_count % eval_every < max(1, n_done):
            eval_result = _eval_policy(mappo, env, eval_seeds_fixed)
            eval_result["approx_episode"] = episode_count
            history["eval_yields"].append(eval_result)
            if verbose:
                print(
                    f"[MAPPO] EVAL ep~{episode_count}: "
                    f"mean_yield={eval_result['mean_yield']:.3f} "
                    f"±{eval_result['std_yield']:.3f}"
                )

    final_eval = _eval_policy(mappo, env, eval_seeds_fixed)
    final_eval["approx_episode"] = episode_count
    history["final_eval"] = final_eval

    with torch.no_grad():
        history["policy_stats"] = {
            "shared_actor_log_std": mappo.actor.log_std.tolist(),
        }

    if verbose:
        print(
            f"[MAPPO] Training done. Final eval yield: "
            f"{final_eval['mean_yield']:.3f} ± {final_eval['std_yield']:.3f}"
        )

    return history
