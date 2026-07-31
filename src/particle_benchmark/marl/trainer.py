"""Training loops for IPPO, MAPPO, MADDPG, COMA, CommNet, and VDN.

Usage
-----
    from particle_benchmark.environment import ParticleEnvConfig
    from particle_benchmark.marl.trainer import train_ippo, train_mappo

    config = ParticleEnvConfig(horizon=67, signal_strength=0.06)
    history = train_ippo(config, n_episodes=500)
    history = train_mappo(config, n_episodes=500)

Each function returns a training-history dict with:
    episode_yields       : list of float  (captures per episode / particle_count)
    eval_yields          : list of dict   (eval results every eval_every episodes)
    policy_stats         : dict           (final log_std etc.)
    algorithm            : str
    checkpoint_episodes  : list of int    (episodes where checkpoints were saved)
    resumed_from         : str | None     (path to checkpoint used to resume)

Checkpointing
-------------
Pass ``checkpoint_dir`` to enable periodic checkpointing.  Checkpoints are
saved as ``{checkpoint_dir}/checkpoint_{episode}.pt`` for on-policy methods
(or ``{checkpoint_dir}/checkpoint_{step}.pt`` for MADDPG).  Each checkpoint
contains the algorithm class name, episode/step count, all state dicts
(networks, optimisers, replay buffer for MADDPG), and the full ``__init__``
kwargs for reconstruction.

Pass ``resume_from`` to load a checkpoint before training begins and continue
from the saved episode/step count.

Requires PyTorch: pip install 'stochastic-particle-system[marl]'
"""

from __future__ import annotations

import time
from pathlib import Path
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
from .maddpg import MADDPG
from .coma import COMA
from .commnet import CommNet
from .vdn import VDN
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


def _eval_policy_with_raw(
    policy: Any,
    env: ParticleCollectorEnv,
    eval_seeds: tuple[int, ...],
) -> dict[str, float]:
    """Evaluate any on-policy algorithm that exposes _get_actions_with_raw."""
    yields = []
    for seed in eval_seeds:
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            actions, _, _, _ = policy._get_actions_with_raw(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        yield_fraction = info["captured_total"] / env.config.particle_count
        yields.append(yield_fraction)
    return {
        "mean_yield": float(np.mean(yields)),
        "std_yield": float(np.std(yields)),
        "min_yield": float(np.min(yields)),
        "max_yield": float(np.max(yields)),
        "n_seeds": len(eval_seeds),
    }


def _eval_maddpg(
    policy: MADDPG,
    env: ParticleCollectorEnv,
    eval_seeds: tuple[int, ...],
) -> dict[str, float]:
    """Evaluate MADDPG in deterministic mode (no exploration noise)."""
    yields = []
    for seed in eval_seeds:
        observations, _ = env.reset(seed=seed)
        done = False
        while not done:
            actions = policy.get_actions(observations, explore=False)
            observations, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        yield_fraction = info["captured_total"] / env.config.particle_count
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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint_onpolicy(
    path: Path,
    algorithm_class: str,
    episode: int,
    init_kwargs: dict[str, Any],
    policy: Any,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save a checkpoint for on-policy algorithms (IPPO, MAPPO, COMA, CommNet, VDN)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "algorithm_class": algorithm_class,
        "episode": episode,
        "init_kwargs": init_kwargs,
    }

    if algorithm_class == "IPPO":
        state["networks"] = [net.state_dict() for net in policy.networks]
        state["optimisers"] = [opt.state_dict() for opt in policy.optimisers]
    elif algorithm_class == "MAPPO":
        state["actor"] = policy.actor.state_dict()
        state["critic"] = policy.critic.state_dict()
        state["actor_optim"] = policy.actor_optim.state_dict()
        state["critic_optim"] = policy.critic_optim.state_dict()
    elif algorithm_class == "COMA":
        state["actor"] = policy.actor.state_dict()
        state["critic"] = policy.critic.state_dict()
        state["actor_optim"] = policy.actor_optim.state_dict()
        state["critic_optim"] = policy.critic_optim.state_dict()
    elif algorithm_class == "CommNet":
        state["net"] = policy.net.state_dict()
        state["optim"] = policy.optim.state_dict()
    elif algorithm_class == "VDN":
        state["actor"] = policy.actor.state_dict()
        state["q_nets"] = [qnet.state_dict() for qnet in policy.q_nets]
        state["optim"] = policy.optim.state_dict()

    if extra_state:
        state.update(extra_state)

    torch.save(state, path)


def _load_checkpoint_onpolicy(
    path: Path,
    algorithm_class: str,
    policy: Any,
) -> int:
    """Load a checkpoint for on-policy algorithms. Returns the saved episode count."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt.get("algorithm_class") == algorithm_class, (
        f"Checkpoint algorithm mismatch: expected {algorithm_class}, "
        f"got {ckpt.get('algorithm_class')}"
    )

    if algorithm_class == "IPPO":
        for net, sd in zip(policy.networks, ckpt["networks"]):
            net.load_state_dict(sd)
        for opt, sd in zip(policy.optimisers, ckpt["optimisers"]):
            opt.load_state_dict(sd)
    elif algorithm_class in ("MAPPO", "COMA"):
        policy.actor.load_state_dict(ckpt["actor"])
        policy.critic.load_state_dict(ckpt["critic"])
        policy.actor_optim.load_state_dict(ckpt["actor_optim"])
        policy.critic_optim.load_state_dict(ckpt["critic_optim"])
    elif algorithm_class == "CommNet":
        policy.net.load_state_dict(ckpt["net"])
        policy.optim.load_state_dict(ckpt["optim"])
    elif algorithm_class == "VDN":
        policy.actor.load_state_dict(ckpt["actor"])
        for qnet, sd in zip(policy.q_nets, ckpt["q_nets"]):
            qnet.load_state_dict(sd)
        policy.optim.load_state_dict(ckpt["optim"])

    return int(ckpt["episode"])


def _maybe_save_checkpoints_onpolicy(
    checkpoint_dir: Path | None,
    checkpoint_every: int,
    prev_episode: int,
    new_episode: int,
    n_episodes: int,
    algorithm_class: str,
    init_kwargs: dict[str, Any],
    policy: Any,
    checkpoint_episodes: list[int],
) -> None:
    """Save checkpoints for any checkpoint boundaries crossed during this update."""
    if checkpoint_dir is None or checkpoint_every <= 0:
        return
    first_boundary = prev_episode // checkpoint_every + 1
    last_boundary = new_episode // checkpoint_every
    for mult in range(first_boundary, last_boundary + 1):
        ep = mult * checkpoint_every
        if ep <= n_episodes:
            path = checkpoint_dir / f"checkpoint_{ep}.pt"
            _save_checkpoint_onpolicy(
                path, algorithm_class, ep, init_kwargs, policy
            )
            checkpoint_episodes.append(ep)


# ---------------------------------------------------------------------------
# IPPO training loop
# ---------------------------------------------------------------------------

def train_ippo(
    env_config: ParticleEnvConfig,
    n_episodes: int = 20_000,
    eval_every: int = 50,
    seed: int = 9001,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    verbose: bool = True,
    checkpoint_every: int = 5_000,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Train IPPO and return training history.

    One training iteration collects ``rollout_steps`` environment steps
    (spanning multiple episodes if needed), then runs a PPO update.
    ``n_episodes`` controls how many *full episode yields* we track, not
    how many PPO updates we perform.

    Parameters
    ----------
    env_config       : environment configuration
    n_episodes       : total training episodes to track yields for
    eval_every       : evaluate on a fixed set of seeds every N tracked episodes
    seed             : base seed; training episodes use seed, seed+1, ...
    rollout_steps    : steps to collect per PPO update
    checkpoint_every : save a checkpoint every N episodes (0 to disable)
    checkpoint_dir   : directory for checkpoints; None disables checkpointing
    resume_from      : path to a checkpoint to resume from; None starts fresh
    """
    env = ParticleCollectorEnv(env_config)
    # Derive obs_dim from one reset.
    init_obs, _ = env.reset(seed=seed)
    obs_dim = compute_obs_dim(init_obs)

    init_kwargs: dict[str, Any] = dict(
        obs_dim=obs_dim,
        n_agents=env_config.collector_count,
        lr=lr,
        gamma=gamma,
        clip_eps=clip_eps,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    ippo = IPPO(**init_kwargs)

    resumed_from: str | None = None
    start_episode: int = 0
    if resume_from is not None:
        start_episode = _load_checkpoint_onpolicy(
            Path(resume_from), "IPPO", ippo
        )
        resumed_from = str(resume_from)

    eval_seeds_fixed = tuple(range(seed + 10000, seed + 10008))

    history: dict[str, Any] = {
        "algorithm": "IPPO",
        "obs_dim": obs_dim,
        "episode_yields": [],
        "eval_yields": [],
        "losses": [],
        "wall_times": [],
        "checkpoint_episodes": [],
        "resumed_from": resumed_from,
    }

    episode_seed = seed + start_episode
    episode_count = start_episode
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
        prev_episode = episode_count
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

        # Checkpointing.
        _maybe_save_checkpoints_onpolicy(
            checkpoint_dir, checkpoint_every,
            prev_episode, episode_count, n_episodes,
            "IPPO", init_kwargs, ippo, history["checkpoint_episodes"],
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
    n_episodes: int = 20_000,
    eval_every: int = 50,
    seed: int = 9001,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    verbose: bool = True,
    checkpoint_every: int = 5_000,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Train MAPPO and return training history.

    Architecture: shared actor (parameter sharing) + centralised critic.

    See :func:`train_ippo` for parameter documentation.
    """
    env = ParticleCollectorEnv(env_config)
    init_obs, _ = env.reset(seed=seed)
    obs_dim = compute_obs_dim(init_obs)

    init_kwargs: dict[str, Any] = dict(
        obs_dim=obs_dim,
        n_agents=env_config.collector_count,
        lr=lr,
        gamma=gamma,
        clip_eps=clip_eps,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    mappo = MAPPO(**init_kwargs)

    resumed_from: str | None = None
    start_episode: int = 0
    if resume_from is not None:
        start_episode = _load_checkpoint_onpolicy(
            Path(resume_from), "MAPPO", mappo
        )
        resumed_from = str(resume_from)

    eval_seeds_fixed = tuple(range(seed + 10000, seed + 10008))

    history: dict[str, Any] = {
        "algorithm": "MAPPO",
        "obs_dim": obs_dim,
        "episode_yields": [],
        "eval_yields": [],
        "losses": [],
        "wall_times": [],
        "checkpoint_episodes": [],
        "resumed_from": resumed_from,
    }

    episode_seed = seed + start_episode
    episode_count = start_episode
    t0 = time.time()

    while episode_count < n_episodes:
        rollout = mappo.collect_rollout(env, n_steps=rollout_steps, seed=episode_seed)
        losses = mappo.update(rollout)
        history["losses"].append(losses)

        dones_any = rollout["dones"][:, 0]
        n_done = int(np.sum(dones_any))
        prev_episode = episode_count
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

        _maybe_save_checkpoints_onpolicy(
            checkpoint_dir, checkpoint_every,
            prev_episode, episode_count, n_episodes,
            "MAPPO", init_kwargs, mappo, history["checkpoint_episodes"],
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


# ---------------------------------------------------------------------------
# MADDPG training loop  (off-policy)
# ---------------------------------------------------------------------------

def train_maddpg(
    env_config: ParticleEnvConfig,
    n_steps: int = 1_340_000,
    eval_every: int = 5_000,
    seed: int = 9001,
    lr_actor: float = 1e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.01,
    batch_size: int = 64,
    buffer_size: int = 100_000,
    noise_std: float = 0.1,
    update_every: int = 1,
    verbose: bool = True,
    checkpoint_every: int = 335_000,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Train MADDPG (off-policy) and return training history.

    Parameters
    ----------
    env_config       : environment configuration
    n_steps          : total environment steps to collect
    eval_every       : evaluate on fixed seeds every N steps
    seed             : base seed for environment resets
    update_every     : call update() every N steps (once buffer is ready)
    checkpoint_every : save a checkpoint every N steps (default 335_000,
                       equivalent to 5_000 episodes × 67 steps)
    checkpoint_dir   : directory for checkpoints; None disables checkpointing
    resume_from      : path to a checkpoint to resume from; None starts fresh
    """
    env = ParticleCollectorEnv(env_config)
    init_obs, _ = env.reset(seed=seed)
    obs_dim = compute_obs_dim(init_obs)

    init_kwargs: dict[str, Any] = dict(
        obs_dim=obs_dim,
        n_agents=env_config.collector_count,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_size=buffer_size,
        noise_std=noise_std,
    )

    maddpg = MADDPG(**init_kwargs)
    maddpg._episode_seed = seed

    resumed_from: str | None = None
    start_step: int = 0
    if resume_from is not None:
        ckpt = torch.load(Path(resume_from), map_location="cpu", weights_only=False)
        assert ckpt.get("algorithm_class") == "MADDPG", (
            f"Checkpoint algorithm mismatch: expected MADDPG, "
            f"got {ckpt.get('algorithm_class')}"
        )
        # Restore actor/critic state dicts.
        for a, sd in zip(maddpg.actors, ckpt["actors"]):
            a.load_state_dict(sd)
        for c, sd in zip(maddpg.critics, ckpt["critics"]):
            c.load_state_dict(sd)
        for a, sd in zip(maddpg.target_actors, ckpt["target_actors"]):
            a.load_state_dict(sd)
        for c, sd in zip(maddpg.target_critics, ckpt["target_critics"]):
            c.load_state_dict(sd)
        for opt, sd in zip(maddpg.actor_optims, ckpt["actor_optims"]):
            opt.load_state_dict(sd)
        for opt, sd in zip(maddpg.critic_optims, ckpt["critic_optims"]):
            opt.load_state_dict(sd)
        # Restore replay buffer state.
        buf = maddpg.buffer
        buf._pos = int(ckpt["buffer_pos"])
        buf._size = int(ckpt["buffer_size_saved"])
        buf._obs[:] = ckpt["buffer_obs"]
        buf._actions[:] = ckpt["buffer_actions"]
        buf._rewards[:] = ckpt["buffer_rewards"]
        buf._next_obs[:] = ckpt["buffer_next_obs"]
        buf._dones[:] = ckpt["buffer_dones"]
        maddpg._episode_seed = int(ckpt.get("episode_seed", seed))
        start_step = int(ckpt["step"])
        resumed_from = str(resume_from)

    eval_seeds_fixed = tuple(range(seed + 10000, seed + 10008))
    checkpoint_steps: list[int] = []

    history: dict[str, Any] = {
        "algorithm": "MADDPG",
        "obs_dim": obs_dim,
        "episode_yields": [],
        "eval_yields": [],
        "losses": [],
        "wall_times": [],
        "checkpoint_episodes": checkpoint_steps,
        "resumed_from": resumed_from,
    }

    t0 = time.time()
    last_eval_step = start_step
    last_ckpt_step = start_step

    for step in range(start_step + 1, n_steps + 1):
        info = maddpg.collect_rollout(env, n_steps=1)

        if maddpg.ready and step % update_every == 0:
            losses = maddpg.update()
            if losses:
                history["losses"].append(losses)

        # Accumulate episode-level yield when an episode finishes.
        if info["n_episodes_done"] > 0:
            yield_frac = info["total_reward"] / (
                env_config.particle_count * info["n_episodes_done"]
            )
            history["episode_yields"].append(yield_frac)
            history["wall_times"].append(time.time() - t0)

            if verbose and len(history["episode_yields"]) % 20 == 0:
                last_loss = history["losses"][-1] if history["losses"] else {}
                print(
                    f"[MADDPG] step={step}/{n_steps}, "
                    f"yield={yield_frac:.3f}, "
                    f"critic_loss={last_loss.get('critic_loss', float('nan')):.4f}"
                )

        # Periodic evaluation.
        if step - last_eval_step >= eval_every:
            eval_result = _eval_maddpg(maddpg, env, eval_seeds_fixed)
            eval_result["step"] = step
            history["eval_yields"].append(eval_result)
            last_eval_step = step
            if verbose:
                print(
                    f"[MADDPG] EVAL step={step}: "
                    f"mean_yield={eval_result['mean_yield']:.3f} "
                    f"±{eval_result['std_yield']:.3f}"
                )

        # Checkpointing.
        if (
            checkpoint_dir is not None
            and checkpoint_every > 0
            and step - last_ckpt_step >= checkpoint_every
        ):
            ckpt_step = (step // checkpoint_every) * checkpoint_every
            ckpt_path = checkpoint_dir / f"checkpoint_{ckpt_step}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            buf = maddpg.buffer
            torch.save(
                {
                    "algorithm_class": "MADDPG",
                    "step": ckpt_step,
                    "init_kwargs": init_kwargs,
                    "actors": [a.state_dict() for a in maddpg.actors],
                    "critics": [c.state_dict() for c in maddpg.critics],
                    "target_actors": [a.state_dict() for a in maddpg.target_actors],
                    "target_critics": [c.state_dict() for c in maddpg.target_critics],
                    "actor_optims": [o.state_dict() for o in maddpg.actor_optims],
                    "critic_optims": [o.state_dict() for o in maddpg.critic_optims],
                    "buffer_pos": buf._pos,
                    "buffer_size_saved": buf._size,
                    "buffer_obs": buf._obs.copy(),
                    "buffer_actions": buf._actions.copy(),
                    "buffer_rewards": buf._rewards.copy(),
                    "buffer_next_obs": buf._next_obs.copy(),
                    "buffer_dones": buf._dones.copy(),
                    "episode_seed": maddpg._episode_seed,
                },
                ckpt_path,
            )
            checkpoint_steps.append(ckpt_step)
            last_ckpt_step = step

    final_eval = _eval_maddpg(maddpg, env, eval_seeds_fixed)
    final_eval["step"] = n_steps
    history["final_eval"] = final_eval
    history["policy_stats"] = {}

    if verbose:
        print(
            f"[MADDPG] Training done. Final eval yield: "
            f"{final_eval['mean_yield']:.3f} ± {final_eval['std_yield']:.3f}"
        )

    return history


# ---------------------------------------------------------------------------
# COMA training loop  (on-policy)
# ---------------------------------------------------------------------------

def train_coma(
    env_config: ParticleEnvConfig,
    n_episodes: int = 20_000,
    eval_every: int = 50,
    seed: int = 9001,
    rollout_steps: int = 2048,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    n_epochs: int = 4,
    batch_size: int = 64,
    n_cf_samples: int = 8,
    verbose: bool = True,
    checkpoint_every: int = 5_000,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Train COMA (on-policy) and return training history.

    Parameters
    ----------
    env_config       : environment configuration
    n_episodes       : approximate training episodes to track
    eval_every       : evaluate every N tracked episodes
    seed             : base seed
    rollout_steps    : environment steps per PPO update
    n_cf_samples     : counterfactual baseline samples per advantage estimate
    checkpoint_every : save a checkpoint every N episodes
    checkpoint_dir   : directory for checkpoints; None disables checkpointing
    resume_from      : path to a checkpoint to resume from; None starts fresh
    """
    env = ParticleCollectorEnv(env_config)
    init_obs, _ = env.reset(seed=seed)
    obs_dim = compute_obs_dim(init_obs)

    init_kwargs: dict[str, Any] = dict(
        obs_dim=obs_dim,
        n_agents=env_config.collector_count,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_cf_samples=n_cf_samples,
    )

    coma = COMA(**init_kwargs)

    resumed_from: str | None = None
    start_episode: int = 0
    if resume_from is not None:
        start_episode = _load_checkpoint_onpolicy(
            Path(resume_from), "COMA", coma
        )
        resumed_from = str(resume_from)

    eval_seeds_fixed = tuple(range(seed + 10000, seed + 10008))

    history: dict[str, Any] = {
        "algorithm": "COMA",
        "obs_dim": obs_dim,
        "episode_yields": [],
        "eval_yields": [],
        "losses": [],
        "wall_times": [],
        "checkpoint_episodes": [],
        "resumed_from": resumed_from,
    }

    episode_seed = seed + start_episode
    episode_count = start_episode
    t0 = time.time()

    while episode_count < n_episodes:
        rollout = coma.collect_rollout(env, n_steps=rollout_steps, seed=episode_seed)
        losses = coma.update(rollout)
        history["losses"].append(losses)

        dones_any = rollout["dones"][:, 0]
        n_done = int(np.sum(dones_any))
        prev_episode = episode_count
        episode_count += max(1, n_done)
        episode_seed += max(1, n_done)

        total_reward = float(rollout["rewards"].sum() / max(1, n_done))
        yield_frac = total_reward / env_config.particle_count
        history["episode_yields"].append(yield_frac)
        history["wall_times"].append(time.time() - t0)

        if verbose and len(history["episode_yields"]) % 10 == 0:
            print(
                f"[COMA] approx episode {episode_count}/{n_episodes}, "
                f"yield={yield_frac:.3f}, "
                f"policy_loss={losses['policy_loss']:.4f}"
            )

        if episode_count % eval_every < max(1, n_done):
            eval_result = _eval_policy_with_raw(coma, env, eval_seeds_fixed)
            eval_result["approx_episode"] = episode_count
            history["eval_yields"].append(eval_result)
            if verbose:
                print(
                    f"[COMA] EVAL ep~{episode_count}: "
                    f"mean_yield={eval_result['mean_yield']:.3f} "
                    f"±{eval_result['std_yield']:.3f}"
                )

        _maybe_save_checkpoints_onpolicy(
            checkpoint_dir, checkpoint_every,
            prev_episode, episode_count, n_episodes,
            "COMA", init_kwargs, coma, history["checkpoint_episodes"],
        )

    final_eval = _eval_policy_with_raw(coma, env, eval_seeds_fixed)
    final_eval["approx_episode"] = episode_count
    history["final_eval"] = final_eval

    with torch.no_grad():
        history["policy_stats"] = {
            "shared_actor_log_std": coma.actor.log_std.tolist(),
        }

    if verbose:
        print(
            f"[COMA] Training done. Final eval yield: "
            f"{final_eval['mean_yield']:.3f} ± {final_eval['std_yield']:.3f}"
        )

    return history


# ---------------------------------------------------------------------------
# CommNet training loop  (on-policy)
# ---------------------------------------------------------------------------

def train_commnet(
    env_config: ParticleEnvConfig,
    n_episodes: int = 20_000,
    eval_every: int = 50,
    seed: int = 9001,
    rollout_steps: int = 2048,
    h_dim: int = 64,
    n_comm_rounds: int = 2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    verbose: bool = True,
    checkpoint_every: int = 5_000,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Train CommNet (on-policy, PPO) and return training history.

    Parameters
    ----------
    env_config       : environment configuration
    n_episodes       : approximate training episodes to track
    eval_every       : evaluate every N tracked episodes
    seed             : base seed
    rollout_steps    : environment steps per PPO update
    h_dim            : hidden / communication vector size
    n_comm_rounds    : K communication rounds per step
    checkpoint_every : save a checkpoint every N episodes
    checkpoint_dir   : directory for checkpoints; None disables checkpointing
    resume_from      : path to a checkpoint to resume from; None starts fresh
    """
    env = ParticleCollectorEnv(env_config)
    init_obs, _ = env.reset(seed=seed)
    obs_dim = compute_obs_dim(init_obs)

    init_kwargs: dict[str, Any] = dict(
        obs_dim=obs_dim,
        h_dim=h_dim,
        n_comm_rounds=n_comm_rounds,
        n_agents=env_config.collector_count,
        lr=lr,
        gamma=gamma,
        clip_eps=clip_eps,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    commnet = CommNet(**init_kwargs)

    resumed_from: str | None = None
    start_episode: int = 0
    if resume_from is not None:
        start_episode = _load_checkpoint_onpolicy(
            Path(resume_from), "CommNet", commnet
        )
        resumed_from = str(resume_from)

    eval_seeds_fixed = tuple(range(seed + 10000, seed + 10008))

    history: dict[str, Any] = {
        "algorithm": "CommNet",
        "obs_dim": obs_dim,
        "episode_yields": [],
        "eval_yields": [],
        "losses": [],
        "wall_times": [],
        "checkpoint_episodes": [],
        "resumed_from": resumed_from,
    }

    episode_seed = seed + start_episode
    episode_count = start_episode
    t0 = time.time()

    while episode_count < n_episodes:
        rollout = commnet.collect_rollout(env, n_steps=rollout_steps, seed=episode_seed)
        losses = commnet.update(rollout)
        history["losses"].append(losses)

        dones_any = rollout["dones"][:, 0]
        n_done = int(np.sum(dones_any))
        prev_episode = episode_count
        episode_count += max(1, n_done)
        episode_seed += max(1, n_done)

        total_reward = float(rollout["rewards"].sum() / max(1, n_done))
        yield_frac = total_reward / env_config.particle_count
        history["episode_yields"].append(yield_frac)
        history["wall_times"].append(time.time() - t0)

        if verbose and len(history["episode_yields"]) % 10 == 0:
            print(
                f"[CommNet] approx episode {episode_count}/{n_episodes}, "
                f"yield={yield_frac:.3f}, "
                f"policy_loss={losses['policy_loss']:.4f}"
            )

        if episode_count % eval_every < max(1, n_done):
            eval_result = _eval_policy_with_raw(commnet, env, eval_seeds_fixed)
            eval_result["approx_episode"] = episode_count
            history["eval_yields"].append(eval_result)
            if verbose:
                print(
                    f"[CommNet] EVAL ep~{episode_count}: "
                    f"mean_yield={eval_result['mean_yield']:.3f} "
                    f"±{eval_result['std_yield']:.3f}"
                )

        _maybe_save_checkpoints_onpolicy(
            checkpoint_dir, checkpoint_every,
            prev_episode, episode_count, n_episodes,
            "CommNet", init_kwargs, commnet, history["checkpoint_episodes"],
        )

    final_eval = _eval_policy_with_raw(commnet, env, eval_seeds_fixed)
    final_eval["approx_episode"] = episode_count
    history["final_eval"] = final_eval

    with torch.no_grad():
        history["policy_stats"] = {
            "comm_log_std": commnet.net.log_std.tolist(),
        }

    if verbose:
        print(
            f"[CommNet] Training done. Final eval yield: "
            f"{final_eval['mean_yield']:.3f} ± {final_eval['std_yield']:.3f}"
        )

    return history


# ---------------------------------------------------------------------------
# VDN training loop  (on-policy)
# ---------------------------------------------------------------------------

def train_vdn(
    env_config: ParticleEnvConfig,
    n_episodes: int = 20_000,
    eval_every: int = 50,
    seed: int = 9001,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 64,
    verbose: bool = True,
    checkpoint_every: int = 5_000,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Train VDN (on-policy, PPO with decomposed values) and return training history.

    Parameters
    ----------
    env_config       : environment configuration
    n_episodes       : approximate training episodes to track
    eval_every       : evaluate every N tracked episodes
    seed             : base seed
    rollout_steps    : environment steps per PPO update
    checkpoint_every : save a checkpoint every N episodes
    checkpoint_dir   : directory for checkpoints; None disables checkpointing
    resume_from      : path to a checkpoint to resume from; None starts fresh
    """
    env = ParticleCollectorEnv(env_config)
    init_obs, _ = env.reset(seed=seed)
    obs_dim = compute_obs_dim(init_obs)

    init_kwargs: dict[str, Any] = dict(
        obs_dim=obs_dim,
        n_agents=env_config.collector_count,
        lr=lr,
        gamma=gamma,
        clip_eps=clip_eps,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    vdn = VDN(**init_kwargs)

    resumed_from: str | None = None
    start_episode: int = 0
    if resume_from is not None:
        start_episode = _load_checkpoint_onpolicy(
            Path(resume_from), "VDN", vdn
        )
        resumed_from = str(resume_from)

    eval_seeds_fixed = tuple(range(seed + 10000, seed + 10008))

    history: dict[str, Any] = {
        "algorithm": "VDN",
        "obs_dim": obs_dim,
        "episode_yields": [],
        "eval_yields": [],
        "losses": [],
        "wall_times": [],
        "checkpoint_episodes": [],
        "resumed_from": resumed_from,
    }

    episode_seed = seed + start_episode
    episode_count = start_episode
    t0 = time.time()

    while episode_count < n_episodes:
        rollout = vdn.collect_rollout(env, n_steps=rollout_steps, seed=episode_seed)
        losses = vdn.update(rollout)
        history["losses"].append(losses)

        dones_any = rollout["dones"][:, 0]
        n_done = int(np.sum(dones_any))
        prev_episode = episode_count
        episode_count += max(1, n_done)
        episode_seed += max(1, n_done)

        total_reward = float(rollout["rewards"].sum() / max(1, n_done))
        yield_frac = total_reward / env_config.particle_count
        history["episode_yields"].append(yield_frac)
        history["wall_times"].append(time.time() - t0)

        if verbose and len(history["episode_yields"]) % 10 == 0:
            print(
                f"[VDN] approx episode {episode_count}/{n_episodes}, "
                f"yield={yield_frac:.3f}, "
                f"policy_loss={losses['policy_loss']:.4f}"
            )

        if episode_count % eval_every < max(1, n_done):
            eval_result = _eval_policy_with_raw(vdn, env, eval_seeds_fixed)
            eval_result["approx_episode"] = episode_count
            history["eval_yields"].append(eval_result)
            if verbose:
                print(
                    f"[VDN] EVAL ep~{episode_count}: "
                    f"mean_yield={eval_result['mean_yield']:.3f} "
                    f"±{eval_result['std_yield']:.3f}"
                )

        _maybe_save_checkpoints_onpolicy(
            checkpoint_dir, checkpoint_every,
            prev_episode, episode_count, n_episodes,
            "VDN", init_kwargs, vdn, history["checkpoint_episodes"],
        )

    final_eval = _eval_policy_with_raw(vdn, env, eval_seeds_fixed)
    final_eval["approx_episode"] = episode_count
    history["final_eval"] = final_eval

    with torch.no_grad():
        history["policy_stats"] = {
            "shared_actor_log_std": vdn.actor.log_std.tolist(),
        }

    if verbose:
        print(
            f"[VDN] Training done. Final eval yield: "
            f"{final_eval['mean_yield']:.3f} ± {final_eval['std_yield']:.3f}"
        )

    return history
