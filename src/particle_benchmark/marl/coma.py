"""Counterfactual Multi-Agent Policy Gradients (COMA) for the particle-collector benchmark.

Reference: Foerster et al., "Counterfactual Multi-Agent Policy Gradients",
AAAI 2018.

Architecture:
  - Shared actor (same parameter-sharing as MAPPO actor).
  - Centralised CounterfactualCritic: Q(s, a_1, …, a_n).
  - Per-agent advantage: A_i = Q(s, a) - (1/K) Σ_k Q(s, a_1, …, ã_i^k, …, a_n)
    where ã_i^k ~ π_i(·|s_i).

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

from .networks import ActorCritic, flatten_all_observations


# ---------------------------------------------------------------------------
# Counterfactual critic
# ---------------------------------------------------------------------------

class CounterfactualCritic(nn.Module):
    """Centralised Q-function that receives global observations and all actions.

    Input dimension: obs_dim * n_agents + 2 * n_agents.
    """

    def __init__(self, global_obs_dim: int, n_agents: int) -> None:
        super().__init__()
        input_dim = global_obs_dim + 2 * n_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, global_obs: torch.Tensor, actions_flat: torch.Tensor
    ) -> torch.Tensor:
        """Return Q-value estimate.

        Parameters
        ----------
        global_obs   : Tensor (..., M * obs_dim)
        actions_flat : Tensor (..., M * 2)
        """
        x = torch.cat([global_obs, actions_flat], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# COMA
# ---------------------------------------------------------------------------

class COMA:
    """Counterfactual Multi-Agent Policy Gradients.

    Parameters
    ----------
    obs_dim      : flat per-agent observation dimension (232 with default config)
    n_agents     : number of collectors (default 4)
    lr_actor     : Adam learning rate for the shared actor
    lr_critic    : Adam learning rate for the counterfactual critic
    gamma        : discount factor
    n_epochs     : PPO-style optimisation epochs per update call
    batch_size   : mini-batch size for SGD
    n_cf_samples : K — number of counterfactual action samples per advantage
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int = 4,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        n_epochs: int = 4,
        batch_size: int = 64,
        n_cf_samples: int = 8,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_cf_samples = n_cf_samples
        self.global_obs_dim = obs_dim * n_agents

        # Shared actor (same interface as MAPPO).
        self.actor = ActorCritic(obs_dim)
        # Centralised counterfactual critic.
        self.critic = CounterfactualCritic(self.global_obs_dim, n_agents)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_actions_with_raw(
        self, observations: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return actions, raw pre-clip actions, log-probs, and values.

        Returns
        -------
        actions     : (M, 2) clipped to unit ball
        raw_actions : (M, 2) pre-clipping samples
        log_probs   : (M,)
        values      : (M,)  from critic with zero-mean action placeholder
        """
        self.actor.eval()
        self.critic.eval()

        flat_obs = flatten_all_observations(observations)  # (M, obs_dim)
        obs_t = torch.from_numpy(flat_obs)  # (M, obs_dim)

        action_means, _ = self.actor.forward(obs_t)  # (M, 2)
        std = self.actor.log_std.exp().expand_as(action_means)
        dist = Normal(action_means, std)
        raw = dist.rsample()                               # (M, 2)
        log_probs = dist.log_prob(raw).sum(dim=-1)        # (M,)
        norm = raw.norm(dim=-1, keepdim=True).clamp(min=1.0)
        actions = raw / norm                               # (M, 2)

        # Value: use critic with current actions as a rough estimate.
        global_obs_t = torch.from_numpy(flat_obs.flatten()).unsqueeze(0)  # (1, M*obs_dim)
        act_flat_t = actions.reshape(1, -1)
        value_scalar = self.critic(global_obs_t, act_flat_t).item()

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

    # ------------------------------------------------------------------
    # Rollout collection (on-policy)
    # ------------------------------------------------------------------

    def collect_rollout(
        self, env: Any, n_steps: int = 2048, seed: int = 0
    ) -> dict[str, Any]:
        """Run the environment for n_steps and return trajectory data.

        Returns a dict with keys:
            obs         : (n_steps, M, obs_dim)
            global_obs  : (n_steps, M * obs_dim)
            raw_actions : (n_steps, M, 2)
            log_probs   : (n_steps, M)
            rewards     : (n_steps, M)
            values      : (n_steps, M)
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
    # COMA update
    # ------------------------------------------------------------------

    def update(self, rollout_data: dict[str, Any]) -> dict[str, float]:
        """Run COMA update: critic trained on MC returns; actor on counterfactual advantage.

        The counterfactual advantage for agent i at timestep t is:
            A_i = Q(s_t, a_t) - (1/K) Σ_k Q(s_t, a_1, …, ã_i^k, …, a_n)
        where ã_i^k ~ π_i(·|o_i_t).
        """
        obs = rollout_data["obs"]               # (T, M, obs_dim)
        global_obs = rollout_data["global_obs"] # (T, M*obs_dim)
        raw_actions = rollout_data["raw_actions"]  # (T, M, 2)
        old_log_probs = rollout_data["log_probs"]  # (T, M)
        rewards = rollout_data["rewards"]           # (T, M)
        dones = rollout_data["dones"]               # (T, M)

        T, M = obs.shape[:2]

        # Monte Carlo returns per agent.
        returns = self._compute_mc_returns(rewards, dones)  # (T, M)

        obs_t = torch.from_numpy(obs.astype(np.float32))            # (T, M, obs_dim)
        global_t = torch.from_numpy(global_obs.astype(np.float32))  # (T, M*obs_dim)
        act_t = torch.from_numpy(raw_actions.astype(np.float32))    # (T, M, 2)
        ret_t = torch.from_numpy(returns.astype(np.float32))        # (T, M)
        old_lp_t = torch.from_numpy(old_log_probs.astype(np.float32))  # (T, M)

        act_flat = act_t.reshape(T, M * 2)  # (T, M*2)

        # --- Critic update: MSE vs mean per-agent MC returns ---
        self.critic.train()
        mean_returns = ret_t.mean(dim=-1)  # (T,)

        critic_loss_total = 0.0
        n_critic_updates = 0
        critic_indices = np.arange(T)
        critic_batch = max(1, self.batch_size // M)

        for _ in range(self.n_epochs):
            np.random.shuffle(critic_indices)
            for start in range(0, T, critic_batch):
                mb = torch.from_numpy(critic_indices[start : start + critic_batch])
                mb_global = global_t[mb]
                mb_act_flat = act_flat[mb]
                mb_ret = mean_returns[mb]

                q_pred = self.critic(mb_global, mb_act_flat)
                loss = nn.functional.mse_loss(q_pred, mb_ret)

                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optim.step()
                critic_loss_total += loss.item()
                n_critic_updates += 1

        # --- Compute counterfactual advantages (no gradient) ---
        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            q_actual = self.critic(global_t, act_flat)  # (T,)

            advantages = torch.zeros(T, M)
            for i in range(M):
                obs_i = obs_t[:, i, :]  # (T, obs_dim)
                action_means, _ = self.actor.forward(obs_i)  # (T, 2)
                std = self.actor.log_std.exp().expand_as(action_means)
                dist_i = Normal(action_means, std)

                baseline = torch.zeros(T)
                for _ in range(self.n_cf_samples):
                    cf_action = dist_i.sample()  # (T, 2)
                    cf_act_flat = act_flat.clone()
                    cf_act_flat[:, i * 2 : i * 2 + 2] = cf_action
                    baseline += self.critic(global_t, cf_act_flat)
                baseline = baseline / self.n_cf_samples

                advantages[:, i] = q_actual - baseline

        # --- Actor update: policy gradient with counterfactual advantage ---
        self.actor.train()

        obs_flat = obs_t.reshape(T * M, self.obs_dim)           # (T*M, obs_dim)
        act_flat_agents = act_t.reshape(T * M, 2)               # (T*M, 2)
        adv_flat = advantages.reshape(T * M)                     # (T*M,)
        old_lp_flat = old_lp_t.reshape(T * M)                   # (T*M,)

        # Normalise advantages.
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        actor_loss_total = 0.0
        entropy_total = 0.0
        n_actor_updates = 0
        agent_indices = np.arange(T * M)

        for _ in range(self.n_epochs):
            np.random.shuffle(agent_indices)
            for start in range(0, T * M, self.batch_size):
                mb = torch.from_numpy(agent_indices[start : start + self.batch_size])
                mb_obs = obs_flat[mb]
                mb_act = act_flat_agents[mb]
                mb_adv = adv_flat[mb]
                mb_old_lp = old_lp_flat[mb]

                new_lp, entropy, _ = self.actor.evaluate_actions(mb_obs, mb_act)

                # Policy gradient (no PPO clip to stay true to COMA paper).
                actor_loss = -(new_lp * mb_adv.detach()).mean()
                entropy_loss = -entropy.mean()
                loss = actor_loss + 0.01 * entropy_loss

                self.actor_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optim.step()
                actor_loss_total += actor_loss.item()
                entropy_total += (-entropy_loss).item()
                n_actor_updates += 1

        return {
            "policy_loss": actor_loss_total / max(1, n_actor_updates),
            "value_loss": critic_loss_total / max(1, n_critic_updates),
            "entropy": entropy_total / max(1, n_actor_updates),
        }

    def _compute_mc_returns(
        self, rewards: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        """Discounted Monte Carlo returns.  rewards/dones: (T, M)."""
        T, M = rewards.shape
        returns = np.zeros_like(rewards)
        running = np.zeros(M, dtype=np.float32)

        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t])
            returns[t] = running

        return returns.astype(np.float32)

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
