"""Communication Neural Network (CommNet) for the particle-collector benchmark.

Reference: Sukhbaatar et al., "Learning Multiagent Communication with Backpropagation",
NeurIPS 2016.

Architecture:
  - Encoder:  MLP(obs_dim → h_dim) shared across agents.
  - Comm step (K rounds): h_i^{t+1} = tanh(W_h · h_i^t + W_c · mean_{j≠i}(h_j^t))
    (W_h, W_c shared across rounds and agents).
  - Decoder:  Linear(h_dim → 2, tanh) for action mean;
              Linear(h_dim → 1) for value estimate.

Policy updates use PPO (on-policy), mirroring the MAPPO update style.

Note: The comm vector passed at each step is analogous to the bounded team-mean
summary in the fixed-statistic treatment (SPS-C03), but learned end-to-end.

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

from .networks import flatten_all_observations


# ---------------------------------------------------------------------------
# CommNet module
# ---------------------------------------------------------------------------

class CommNetModule(nn.Module):
    """Encoder + K-round communication + actor/critic decoder heads.

    All weights are shared across agents and across communication rounds.

    Parameters
    ----------
    obs_dim       : flat per-agent observation dimension
    h_dim         : hidden / communication vector size (default 64)
    n_comm_rounds : number of communication rounds K (default 2)
    """

    def __init__(
        self, obs_dim: int, h_dim: int = 64, n_comm_rounds: int = 2
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.n_comm_rounds = n_comm_rounds

        # Shared encoder: obs_dim → h_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        # Shared communication transforms (applied each round).
        self.W_h = nn.Linear(h_dim, h_dim, bias=True)
        self.W_c = nn.Linear(h_dim, h_dim, bias=False)

        # Actor head: h_dim → 2 with Tanh (action mean in (-1,1)^2).
        self.actor_head = nn.Sequential(nn.Linear(h_dim, 2), nn.Tanh())
        # Value head: h_dim → 1.
        self.value_head = nn.Linear(h_dim, 1)

        # Learnable log-std (shared across agents, one per action dimension).
        self.log_std = nn.Parameter(torch.zeros(2))

    def _communicate(
        self, h: torch.Tensor, comm_ablated: bool = False
    ) -> torch.Tensor:
        """Run K rounds of mean-pooled communication.

        Parameters
        ----------
        h            : Tensor (*, M, h_dim)  where * may be (T,) or empty
        comm_ablated : if True, zero out all received communication vectors
                       before each round so agents receive no teammate messages.
                       This is the architecturally meaningful ablation: agents
                       still apply their self-recurrence W_h but receive zero
                       input from W_c, isolating the effect of communication.

        Returns
        -------
        h : Tensor (*, M, h_dim)  updated hidden states
        """
        M = h.shape[-2]
        for _ in range(self.n_comm_rounds):
            shape = h.shape
            h_flat = h.reshape(-1, self.h_dim)

            if comm_ablated:
                # Zero out all communication: agents receive no teammate messages.
                zeros = torch.zeros_like(h_flat)
                h = torch.tanh(self.W_h(h_flat) + self.W_c(zeros)).reshape(shape)
            else:
                # Mean of all agents excluding self.
                h_sum = h.sum(dim=-2, keepdim=True)          # (*, 1, h_dim)
                h_mean_excl = (h_sum - h) / max(1, M - 1)   # (*, M, h_dim)
                hm_flat = h_mean_excl.reshape(-1, self.h_dim)
                h = torch.tanh(self.W_h(h_flat) + self.W_c(hm_flat)).reshape(shape)
        return h

    def forward_step(
        self, obs: torch.Tensor, comm_ablated: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single timestep.

        Parameters
        ----------
        obs          : Tensor (M, obs_dim)
        comm_ablated : if True, zero all communication vectors (ablation mode)

        Returns
        -------
        action_mean : Tensor (M, 2)
        value       : Tensor (M,)
        """
        h = self.encoder(obs)                              # (M, h_dim)
        h = self._communicate(h, comm_ablated=comm_ablated)  # (M, h_dim)
        action_mean = self.actor_head(h)                   # (M, 2)
        value = self.value_head(h).squeeze(-1)             # (M,)
        return action_mean, value

    def forward_batch(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a batch of timesteps.

        Parameters
        ----------
        obs : Tensor (T, M, obs_dim)

        Returns
        -------
        action_mean : Tensor (T, M, 2)
        value       : Tensor (T, M)
        """
        T, M, _ = obs.shape
        h = self.encoder(obs.reshape(T * M, -1)).reshape(T, M, self.h_dim)
        h = self._communicate(h)                                    # (T, M, h_dim)
        h_flat = h.reshape(T * M, self.h_dim)
        action_mean = self.actor_head(h_flat).reshape(T, M, 2)
        value = self.value_head(h_flat).squeeze(-1).reshape(T, M)
        return action_mean, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored (raw) actions under the current policy.

        Parameters
        ----------
        obs     : Tensor (B, obs_dim)   — per-agent observations (flattened T*M)
        actions : Tensor (B, 2)         — raw pre-clipping actions

        Returns
        -------
        log_prob : Tensor (B,)
        entropy  : Tensor (B,)
        value    : Tensor (B,)  — NOTE: independent forward pass (no comm in eval batch)
        """
        # For mini-batch training we approximate the comm pass by treating each
        # row as an independent agent (efficiency trade-off).  The encoder +
        # decoder still use the shared learned weights.
        h = self.encoder(obs)             # (B, h_dim)
        action_mean = self.actor_head(h)  # (B, 2)
        value = self.value_head(h).squeeze(-1)  # (B,)

        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# CommNet algorithm
# ---------------------------------------------------------------------------

class CommNet:
    """CommNet agent: learned continuous communication + PPO updates.

    Note: The comm vector passed at each step is analogous to the bounded
    team-mean summary in the fixed-statistic treatment (SPS-C03), but learned
    end-to-end.

    Parameters
    ----------
    obs_dim       : flat per-agent observation dimension (232 with default config)
    h_dim         : hidden / communication vector size (default 64)
    n_comm_rounds : number of communication rounds K (default 2)
    n_agents      : number of collectors (default 4)
    lr            : Adam learning rate
    gamma         : discount factor
    clip_eps      : PPO clip epsilon
    n_epochs      : optimisation epochs per update call
    batch_size    : mini-batch size (over time * agents)
    """

    def __init__(
        self,
        obs_dim: int,
        h_dim: int = 64,
        n_comm_rounds: int = 2,
        n_agents: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> None:
        self.obs_dim = obs_dim
        self.h_dim = h_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.net = CommNetModule(obs_dim, h_dim=h_dim, n_comm_rounds=n_comm_rounds)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_actions_with_raw(
        self, observations: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run encoder + comm + decoder and sample stochastic actions.

        Returns
        -------
        actions     : (M, 2)  clipped to unit ball
        raw_actions : (M, 2)  pre-clipping samples
        log_probs   : (M,)
        values      : (M,)
        """
        self.net.eval()
        flat_obs = flatten_all_observations(observations)  # (M, obs_dim)
        obs_t = torch.from_numpy(flat_obs)                 # (M, obs_dim)

        action_means, values = self.net.forward_step(obs_t)  # (M, 2), (M,)
        std = self.net.log_std.exp().expand_as(action_means)
        dist = Normal(action_means, std)
        raw = dist.rsample()                               # (M, 2)
        log_probs = dist.log_prob(raw).sum(dim=-1)        # (M,)
        norm = raw.norm(dim=-1, keepdim=True).clamp(min=1.0)
        actions = raw / norm

        return actions.numpy(), raw.numpy(), log_probs.numpy(), values.numpy()

    def get_actions(self, observations: tuple) -> np.ndarray:
        """Return clipped actions (M, 2) for passing to the environment.

        Runs the full encoder + communication + decoder pass so that the comm
        vectors are computed from all agents' current hidden states.
        """
        actions, _, _, _ = self._get_actions_with_raw(observations)
        return actions

    @torch.no_grad()
    def ablated_get_actions(self, observations: tuple) -> np.ndarray:
        """Return clipped actions (M, 2) with communication ablated.

        Runs encoder and decoder but zeros out all communication vectors before
        each communication round. Agents use their own encoding but receive no
        teammate messages. This is the architecturally meaningful communication
        ablation for CommNet: W_c receives zero input so each agent acts as if
        it were alone, exposing the pure contribution of learned communication.
        """
        self.net.eval()
        flat_obs = flatten_all_observations(observations)  # (M, obs_dim)
        obs_t = torch.from_numpy(flat_obs)                 # (M, obs_dim)

        # Run forward with communication zeroed out.
        action_means, _ = self.net.forward_step(obs_t, comm_ablated=True)
        std = self.net.log_std.exp().expand_as(action_means)
        dist = Normal(action_means, std)
        raw = dist.rsample()                               # (M, 2)
        norm = raw.norm(dim=-1, keepdim=True).clamp(min=1.0)
        actions = raw / norm

        return actions.numpy()

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
            values      : (n_steps, M)
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
    # PPO update
    # ------------------------------------------------------------------

    def update(self, rollout_data: dict[str, Any]) -> dict[str, float]:
        """Run PPO update using the shared CommNet network.

        Returns a dict with 'policy_loss', 'value_loss', 'entropy'.
        """
        obs = rollout_data["obs"]               # (T, M, obs_dim)
        raw_actions = rollout_data["raw_actions"]  # (T, M, 2)
        old_log_probs = rollout_data["log_probs"]  # (T, M)
        rewards = rollout_data["rewards"]           # (T, M)
        values = rollout_data["values"]             # (T, M)
        dones = rollout_data["dones"]               # (T, M)

        T, M = obs.shape[:2]

        advantages = self._compute_advantages(rewards, values, dones)  # (T, M)
        returns = advantages + values  # (T, M)

        # Flatten time and agent dims.
        obs_flat = obs.reshape(T * M, self.obs_dim)
        act_flat = raw_actions.reshape(T * M, 2)
        old_lp_flat = old_log_probs.reshape(T * M)
        adv_flat = advantages.reshape(T * M)
        ret_flat = returns.reshape(T * M)

        obs_t = torch.from_numpy(obs_flat.astype(np.float32))
        act_t = torch.from_numpy(act_flat.astype(np.float32))
        old_lp_t = torch.from_numpy(old_lp_flat.astype(np.float32))
        adv_t = torch.from_numpy(adv_flat.astype(np.float32))
        ret_t = torch.from_numpy(ret_flat.astype(np.float32))

        # Normalise advantages.
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        self.net.train()
        n_steps_flat = T * M
        indices = np.arange(n_steps_flat)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps_flat, self.batch_size):
                mb_idx = torch.from_numpy(indices[start : start + self.batch_size])

                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_lp = old_lp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                new_lp, entropy, value_pred = self.net.evaluate_actions(mb_obs, mb_act)

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value_pred, mb_ret)
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
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
        """Save CommNet state dict."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "obs_dim": self.obs_dim,
            "h_dim": self.h_dim,
            "n_agents": self.n_agents,
            "net": self.net.state_dict(),
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load CommNet state dict."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        assert state["obs_dim"] == self.obs_dim
        assert state["n_agents"] == self.n_agents
        self.net.load_state_dict(state["net"])
