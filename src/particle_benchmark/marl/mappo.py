"""Multi-Agent PPO (MAPPO) baseline for the particle-collector benchmark.

One actor network is *shared* across all agents (parameter sharing).
The critic is *centralised*: it receives the concatenated flat observations
of every agent (global_obs_dim = obs_dim * n_agents) and produces a single
value estimate used to compute advantages for all agents.

Requires PyTorch: pip install 'stochastic-particle-system[marl]'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except ImportError:
    raise ImportError(
        "MARL baselines require PyTorch: pip install 'stochastic-particle-system[marl]'"
    )

from .networks import ActorCritic, CentralizedCritic, flatten_all_observations


class MAPPO:
    """Multi-Agent PPO with centralised critic and shared actor weights.

    Parameters
    ----------
    obs_dim   : flat per-agent observation dimension (232 with default config)
    n_agents  : number of collectors (default 4)
    lr        : Adam learning rate
    gamma     : discount factor
    gae_lambda: GAE lambda for advantage estimation
    clip_eps  : PPO clip epsilon
    n_epochs  : number of optimisation epochs per update call
    batch_size: mini-batch size for SGD (over time steps * agents)
    vf_coef   : value-function loss coefficient
    ent_coef  : entropy bonus coefficient
    """

    execution_time_communication = False
    training_time_centralization = True

    def __init__(
        self,
        obs_dim: int,
        n_agents: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.global_obs_dim = obs_dim * n_agents

        # Shared actor (same weights for all agents).
        self.actor = ActorCritic(obs_dim)
        # Centralised critic (global observation).
        self.critic = CentralizedCritic(self.global_obs_dim)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_actions_with_raw(
        self, observations: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return actions, raw pre-clip actions, log-probs, and centralized values.

        Returns
        -------
        actions      : (M, 2) clipped to unit ball
        raw_actions  : (M, 2) pre-clipping samples
        log_probs    : (M,)
        values       : (M,)  from centralised critic (same for all agents)
        """
        self.actor.eval()
        self.critic.eval()

        flat_obs = flatten_all_observations(observations)  # (M, obs_dim)
        global_obs_t = torch.from_numpy(
            flat_obs.flatten().astype(np.float32)
        ).unsqueeze(0)  # (1, M*obs_dim)
        value_scalar = self.critic(global_obs_t).item()  # same for all agents

        obs_t = torch.from_numpy(flat_obs)  # (M, obs_dim)
        action_means, _ = self.actor.forward(obs_t)  # (M, 2)
        std = self.actor.log_std.exp().expand_as(action_means)
        dist = Normal(action_means, std)
        raw = dist.rsample()                                    # (M, 2)
        log_probs = dist.log_prob(raw).sum(dim=-1)              # (M,)
        norm = raw.norm(dim=-1, keepdim=True).clamp(min=1.0)
        actions = raw / norm

        return (
            actions.numpy(),
            raw.numpy(),
            log_probs.numpy(),
            np.full(self.n_agents, value_scalar, dtype=np.float32),
        )

    def get_actions(self, observations: tuple) -> np.ndarray:
        """Return clipped actions (M, 2) for passing to the environment."""
        actions, _, _, _ = self._get_actions_with_raw(observations)
        return actions

    def ablated_get_actions(self, observations: tuple) -> np.ndarray:
        """MAPPO uses CTDE, not an execution-time message channel."""
        raise NotImplementedError(
            "MAPPO's centralized critic is training-time CTDE. Its actor has "
            "no execution-time message to ablate."
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self, env: Any, n_steps: int = 2048, seed: int = 0
    ) -> dict[str, Any]:
        """Run the environment for n_steps and return trajectory data.

        Returns a dict with keys:
            obs         : (n_steps, M, obs_dim)
            global_obs  : (n_steps, M*obs_dim)    centralized observations
            raw_actions : (n_steps, M, 2)
            log_probs   : (n_steps, M)
            rewards     : (n_steps, M)
            values      : (n_steps, M)             from centralized critic
            dones       : (n_steps, M)
        """
        obs_buf = np.zeros((n_steps, self.n_agents, self.obs_dim), dtype=np.float32)
        global_obs_buf = np.zeros((n_steps, self.global_obs_dim), dtype=np.float32)
        raw_action_buf = np.zeros((n_steps, self.n_agents, 2), dtype=np.float32)
        log_prob_buf = np.zeros((n_steps, self.n_agents), dtype=np.float32)
        reward_buf = np.zeros((n_steps, self.n_agents), dtype=np.float32)
        value_buf = np.zeros((n_steps, self.n_agents), dtype=np.float32)
        done_buf = np.zeros((n_steps, self.n_agents), dtype=np.float32)

        episode_seed = seed
        observations, _ = env.reset(seed=episode_seed)

        for step in range(n_steps):
            flat_obs = flatten_all_observations(observations)  # (M, obs_dim)
            actions, raw_actions, log_probs, values = self._get_actions_with_raw(
                observations
            )

            obs_buf[step] = flat_obs
            global_obs_buf[step] = flat_obs.flatten()
            raw_action_buf[step] = raw_actions
            log_prob_buf[step] = log_probs
            value_buf[step] = values

            next_obs, reward, terminated, truncated, info = env.step(actions)

            per_agent_reward = np.array(reward, dtype=np.float32)
            reward_buf[step] = per_agent_reward

            done = terminated or truncated
            done_buf[step] = float(done)

            if done:
                episode_seed += 1
                observations, _ = env.reset(seed=episode_seed)
            else:
                observations = next_obs

        return {
            "obs": obs_buf,
            "global_obs": global_obs_buf,
            "raw_actions": raw_action_buf,
            "log_probs": log_prob_buf,
            "rewards": reward_buf,
            "values": value_buf,
            "dones": done_buf,
        }

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, rollout_data: dict[str, Any]) -> dict[str, float]:
        """Run MAPPO update: shared actor + centralised critic.

        The actor is updated using per-agent advantages (from centralized
        value estimates), and the critic is updated to predict the mean
        per-agent return.
        """
        obs = rollout_data["obs"]               # (T, M, obs_dim)
        global_obs = rollout_data["global_obs"] # (T, M*obs_dim)
        raw_actions = rollout_data["raw_actions"]  # (T, M, 2)
        old_log_probs = rollout_data["log_probs"]  # (T, M)
        rewards = rollout_data["rewards"]           # (T, M)
        values = rollout_data["values"]             # (T, M)
        dones = rollout_data["dones"]               # (T, M)

        T, M = obs.shape[:2]

        advantages = self._compute_advantages(rewards, values, dones)  # (T, M)
        returns = advantages + values  # (T, M)

        # Flatten time and agent dimensions for joint training.
        # Actor sees (T*M, obs_dim); critic sees (T, M*obs_dim).
        obs_flat = obs.reshape(T * M, self.obs_dim)              # (T*M, obs_dim)
        act_flat = raw_actions.reshape(T * M, 2)                 # (T*M, 2)
        old_lp_flat = old_log_probs.reshape(T * M)               # (T*M,)
        adv_flat = advantages.reshape(T * M)                      # (T*M,)
        ret_flat_agent = returns.reshape(T * M)                   # (T*M,) per-agent

        # Centralised returns: average across agents.
        ret_centralized = returns.mean(axis=1)   # (T,) one per timestep

        obs_flat_t = torch.from_numpy(obs_flat.astype(np.float32))
        act_flat_t = torch.from_numpy(act_flat.astype(np.float32))
        old_lp_t = torch.from_numpy(old_lp_flat.astype(np.float32))
        adv_t = torch.from_numpy(adv_flat.astype(np.float32))
        ret_c_t = torch.from_numpy(ret_centralized.astype(np.float32))
        global_obs_t = torch.from_numpy(global_obs.astype(np.float32))  # (T, global)

        # Normalise advantages.
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n_agent_steps = T * M
        n_critic_steps = T
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        self.actor.train()
        self.critic.train()

        agent_indices = np.arange(n_agent_steps)
        critic_indices = np.arange(n_critic_steps)

        for _ in range(self.n_epochs):
            np.random.shuffle(agent_indices)
            np.random.shuffle(critic_indices)

            # Actor updates (over flattened agent*time).
            for start in range(0, n_agent_steps, self.batch_size):
                mb_idx = agent_indices[start : start + self.batch_size]
                mb_idx_t = torch.from_numpy(mb_idx)

                mb_obs = obs_flat_t[mb_idx_t]
                mb_act = act_flat_t[mb_idx_t]
                mb_old_lp = old_lp_t[mb_idx_t]
                mb_adv = adv_t[mb_idx_t]

                new_lp, entropy, _ = self.actor.evaluate_actions(mb_obs, mb_act)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                actor_loss = policy_loss + self.ent_coef * entropy_loss

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optim.step()

                total_policy_loss += policy_loss.item()
                total_entropy += (-entropy_loss).item()
                n_updates += 1

            # Critic updates (over timesteps).
            critic_batch = max(1, self.batch_size // M)
            for start in range(0, n_critic_steps, critic_batch):
                mb_idx = critic_indices[start : start + critic_batch]
                mb_idx_t = torch.from_numpy(mb_idx)

                mb_global = global_obs_t[mb_idx_t]
                mb_ret = ret_c_t[mb_idx_t]

                value_pred = self.critic(mb_global)
                value_loss = nn.functional.mse_loss(value_pred, mb_ret)

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()

                total_value_loss += value_loss.item()

        n_critic_updates = self.n_epochs * max(1, n_critic_steps // max(1, self.batch_size // M))
        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_critic_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def _compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """GAE advantage estimation.  All arrays: (T, M)."""
        T, M = rewards.shape
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(M, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = values[-1] * (1.0 - dones[-1])
            else:
                next_val = values[t + 1] * (1.0 - dones[t + 1])
            delta = rewards[t] + self.gamma * next_val - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        return advantages.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save actor and critic state dicts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "obs_dim": self.obs_dim,
            "n_agents": self.n_agents,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load actor and critic state dicts."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        assert state["obs_dim"] == self.obs_dim
        assert state["n_agents"] == self.n_agents
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
