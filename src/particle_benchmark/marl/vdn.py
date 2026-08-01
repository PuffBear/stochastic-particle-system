"""Experimental continuous-action value-decomposition-style method.

Reference: Sunehag et al., "Value-Decomposition Networks For Cooperative Multi-Agent
Learning", AAMAS 2018.

VDN decomposes the joint value as Q_tot = Σ_i Q_i (identity mixing).  For
continuous actions an actor-critic variant is used:
  - Shared stochastic actor (parameter sharing across agents).
  - Per-agent independent Q-net Q_i(obs_i) for local value estimation.
  - VDNMixer sums local values: Q_tot = Σ Q_i (no learnable parameters).
  - Advantage for agent i: A_i = Q_i − (Q_tot / M), i.e. local value minus
    average contribution; optimised with PPO-style clipped objective.

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
# VDN components
# ---------------------------------------------------------------------------

class IndependentQNet(nn.Module):
    """Per-agent local value estimator Q_i(obs_i).

    Architecture: Linear(obs_dim, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1)
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return local value estimate.  obs: (..., obs_dim)."""
        return self.net(obs).squeeze(-1)


class VDNMixer(nn.Module):
    """Identity value mixer: Q_tot = Σ_i Q_i.  No learnable parameters."""

    def forward(self, q_values: torch.Tensor) -> torch.Tensor:
        """Sum local Q values to get joint Q_tot.

        Parameters
        ----------
        q_values : Tensor (..., M)

        Returns
        -------
        q_tot : Tensor (...)
        """
        return q_values.sum(dim=-1)


# ---------------------------------------------------------------------------
# VDN algorithm
# ---------------------------------------------------------------------------

class VDN:
    """Continuous-action value-decomposition-style prototype.

    This actor-critic adaptation is not canonical discrete-action VDN and must
    be reported as ``continuous-VD-style`` pending algorithmic validation.

    Parameters
    ----------
    obs_dim   : flat per-agent observation dimension (232 with default config)
    n_agents  : number of collectors (default 4)
    lr        : Adam learning rate (shared for actor and Q-nets)
    gamma     : discount factor
    clip_eps  : PPO clip epsilon for actor update
    n_epochs  : optimisation epochs per update call
    batch_size: mini-batch size (over time * agents)
    """

    report_label = "continuous-VD-style (experimental)"
    execution_time_communication = False
    training_time_centralization = True

    def __init__(
        self,
        obs_dim: int,
        n_agents: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Shared stochastic actor (reuses ActorCritic backbone; critic_head unused).
        self.actor = ActorCritic(obs_dim)

        # Per-agent independent Q-nets.
        self.q_nets = nn.ModuleList(
            [IndependentQNet(obs_dim) for _ in range(n_agents)]
        )

        # VDN mixer (no parameters).
        self.mixer = VDNMixer()

        # Single optimiser for actor + all Q-nets.
        all_params = list(self.actor.parameters()) + list(self.q_nets.parameters())
        self.optim = optim.Adam(all_params, lr=lr)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_actions_with_raw(
        self, observations: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions from the shared actor; collect per-agent Q values.

        Returns
        -------
        actions     : (M, 2)  clipped to unit ball
        raw_actions : (M, 2)  pre-clipping samples
        log_probs   : (M,)
        values      : (M,)    per-agent Q_i estimates
        """
        self.actor.eval()
        for qnet in self.q_nets:
            qnet.eval()

        flat_obs = flatten_all_observations(observations)  # (M, obs_dim)
        obs_t = torch.from_numpy(flat_obs)                 # (M, obs_dim)

        action_means, _ = self.actor.forward(obs_t)  # (M, 2)
        std = self.actor.log_std.exp().expand_as(action_means)
        dist = Normal(action_means, std)
        raw = dist.rsample()                               # (M, 2)
        log_probs = dist.log_prob(raw).sum(dim=-1)        # (M,)
        norm = raw.norm(dim=-1, keepdim=True).clamp(min=1.0)
        actions = raw / norm                               # (M, 2)

        # Per-agent local Q values.
        values = torch.stack(
            [self.q_nets[i](obs_t[i : i + 1]).squeeze() for i in range(self.n_agents)]
        )  # (M,)

        return actions.numpy(), raw.numpy(), log_probs.numpy(), values.numpy()

    def get_actions(self, observations: tuple) -> np.ndarray:
        """Return clipped actions (M, 2) for passing to the environment."""
        actions, _, _, _ = self._get_actions_with_raw(observations)
        return actions

    def ablated_get_actions(self, observations: tuple) -> np.ndarray:
        """Value decomposition is CTDE, not communication."""
        raise NotImplementedError(
            "continuous-VD-style has no execution-time message channel; an "
            "identity actor comparison cannot support a communication claim."
        )

    # ------------------------------------------------------------------
    # Rollout collection (on-policy)
    # ------------------------------------------------------------------

    def collect_rollout(
        self, env: Any, n_steps: int = 2048, seed: int = 0
    ) -> dict[str, Any]:
        """Run the environment for n_steps and return trajectory data.

        Returns a dict with keys:
            obs         : (n_steps, M, obs_dim)
            raw_actions : (n_steps, M, 2)
            log_probs   : (n_steps, M)
            rewards     : (n_steps, M)
            values      : (n_steps, M)  per-agent Q_i
            dones       : (n_steps, M)
        """
        obs_buf = np.zeros((n_steps, self.n_agents, self.obs_dim), dtype=np.float32)
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
            "raw_actions": raw_action_buf,
            "log_probs": log_prob_buf,
            "rewards": reward_buf,
            "values": value_buf,
            "dones": done_buf,
        }

    # ------------------------------------------------------------------
    # VDN update
    # ------------------------------------------------------------------

    def update(self, rollout_data: dict[str, Any]) -> dict[str, float]:
        """Run VDN update: fit per-agent Q-nets and update shared actor.

        Value target for Q_i: per-agent discounted return r_i.
        Advantage for agent i: A_i = Q_i − (Q_tot / M)
                                    = Q_i − mean_j Q_j
        (i.e. how much more agent i contributes than average).

        Returns dict with 'policy_loss', 'value_loss', 'entropy'.
        """
        obs = rollout_data["obs"]               # (T, M, obs_dim)
        raw_actions = rollout_data["raw_actions"]  # (T, M, 2)
        old_log_probs = rollout_data["log_probs"]  # (T, M)
        rewards = rollout_data["rewards"]           # (T, M)
        values = rollout_data["values"]             # (T, M)  per-agent Q_i
        dones = rollout_data["dones"]               # (T, M)

        T, M = obs.shape[:2]

        # Per-agent GAE advantages and returns.
        advantages = self._compute_advantages(rewards, values, dones)  # (T, M)
        returns = advantages + values  # (T, M)

        # Flatten for mini-batch training.
        obs_flat = obs.reshape(T * M, self.obs_dim)
        act_flat = raw_actions.reshape(T * M, 2)
        old_lp_flat = old_log_probs.reshape(T * M)
        adv_flat = advantages.reshape(T * M)
        ret_flat = returns.reshape(T * M)

        # Track which agent each element belongs to (for Q-net routing).
        agent_idx_flat = np.tile(np.arange(M), T)  # (T*M,) repeating 0,1,…,M-1

        obs_t = torch.from_numpy(obs_flat.astype(np.float32))
        act_t = torch.from_numpy(act_flat.astype(np.float32))
        old_lp_t = torch.from_numpy(old_lp_flat.astype(np.float32))
        adv_t = torch.from_numpy(adv_flat.astype(np.float32))
        ret_t = torch.from_numpy(ret_flat.astype(np.float32))

        # Normalise advantages.
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        self.actor.train()
        for qnet in self.q_nets:
            qnet.train()

        n_steps_flat = T * M
        indices = np.arange(n_steps_flat)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps_flat, self.batch_size):
                mb_idx = indices[start : start + self.batch_size]
                mb_idx_t = torch.from_numpy(mb_idx)

                mb_obs = obs_t[mb_idx_t]
                mb_act = act_t[mb_idx_t]
                mb_old_lp = old_lp_t[mb_idx_t]
                mb_adv = adv_t[mb_idx_t]
                mb_ret = ret_t[mb_idx_t]
                mb_agent = agent_idx_flat[mb_idx]  # (mb,) agent indices

                # Actor PPO update (shared).
                new_lp, entropy, _ = self.actor.evaluate_actions(mb_obs, mb_act)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                # Per-agent Q-net value loss: route each sample to its Q-net.
                value_pred = torch.zeros(len(mb_idx))
                for i in range(self.n_agents):
                    mask = torch.from_numpy((mb_agent == i))
                    if mask.any():
                        value_pred[mask] = self.q_nets[i](mb_obs[mask])
                value_loss = nn.functional.mse_loss(value_pred, mb_ret)

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.optim.zero_grad()
                loss.backward()
                all_params = list(self.actor.parameters()) + list(self.q_nets.parameters())
                nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
                self.optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
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
            last_gae = delta + self.gamma * 0.95 * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        return advantages.astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save actor and per-agent Q-net state dicts."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "obs_dim": self.obs_dim,
            "n_agents": self.n_agents,
            "actor": self.actor.state_dict(),
            "q_nets": [qnet.state_dict() for qnet in self.q_nets],
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load actor and per-agent Q-net state dicts."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        assert state["obs_dim"] == self.obs_dim
        assert state["n_agents"] == self.n_agents
        self.actor.load_state_dict(state["actor"])
        for qnet, sd in zip(self.q_nets, state["q_nets"]):
            qnet.load_state_dict(sd)
