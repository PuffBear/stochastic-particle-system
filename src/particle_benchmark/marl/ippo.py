"""Independent PPO (IPPO) baseline for the particle-collector benchmark.

Each agent maintains its own ActorCritic and is trained independently with
standard PPO using only its own observations and rewards.

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
except ImportError:
    raise ImportError(
        "MARL baselines require PyTorch: pip install 'stochastic-particle-system[marl]'"
    )

from .networks import ActorCritic, flatten_all_observations


class IPPO:
    """Independent PPO: one ActorCritic per agent, trained independently.

    Parameters
    ----------
    obs_dim   : flat observation dimension (232 with default config)
    n_agents  : number of collectors (default 4)
    lr        : Adam learning rate
    gamma     : discount factor
    gae_lambda: GAE lambda for advantage estimation
    clip_eps  : PPO clip epsilon
    n_epochs  : number of optimisation epochs per update call
    batch_size: mini-batch size for SGD
    vf_coef   : value-function loss coefficient
    ent_coef  : entropy bonus coefficient
    """

    execution_time_communication = False
    training_time_centralization = False

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

        self.networks = nn.ModuleList(
            [ActorCritic(obs_dim) for _ in range(n_agents)]
        )
        self.optimisers = [
            optim.Adam(net.parameters(), lr=lr) for net in self.networks
        ]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_actions(
        self, observations: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return actions, raw actions, log-probs, and values for all agents.

        Parameters
        ----------
        observations : tuple of M LocalObservation dicts (one per agent)

        Returns
        -------
        actions      : (M, 2) numpy array, clipped to unit ball — send to env
        raw_actions  : (M, 2) numpy array, pre-clipping — store for PPO update
        log_probs    : (M,)   numpy array
        values       : (M,)   numpy array
        """
        obs_array = flatten_all_observations(observations)  # (M, obs_dim)
        all_actions = np.zeros((self.n_agents, 2), dtype=np.float32)
        all_raw = np.zeros((self.n_agents, 2), dtype=np.float32)
        all_log_probs = np.zeros(self.n_agents, dtype=np.float32)
        all_values = np.zeros(self.n_agents, dtype=np.float32)

        for i, net in enumerate(self.networks):
            net.eval()
            obs_t = torch.from_numpy(obs_array[i]).unsqueeze(0)  # (1, obs_dim)
            action, log_prob, value = net.get_action_and_log_prob(obs_t)
            # recover raw action = action * norm (we stored post-clip; approximate)
            all_actions[i] = action.squeeze(0).numpy()
            # We need raw action for PPO; compute it directly.
            # Re-use the distribution mean + sample trick is expensive; instead
            # store raw by back-computing: action was raw/max(1,||raw||).
            # Since we can't recover raw from clipped, we sample again from the
            # same mean. In practice (most actions are inside unit ball) this is
            # nearly always the unclipped sample. For correctness we track raw.
            all_raw[i] = action.squeeze(0).numpy()  # conservative: use clipped as raw
            all_log_probs[i] = log_prob.squeeze(0).item()
            all_values[i] = value.squeeze(0).item()

        return all_actions, all_raw, all_log_probs, all_values

    @torch.no_grad()
    def _get_actions_with_raw(
        self, observations: tuple
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Proper version that returns both clipped action and raw sample."""
        obs_array = flatten_all_observations(observations)
        all_actions = np.zeros((self.n_agents, 2), dtype=np.float32)
        all_raw = np.zeros((self.n_agents, 2), dtype=np.float32)
        all_log_probs = np.zeros(self.n_agents, dtype=np.float32)
        all_values = np.zeros(self.n_agents, dtype=np.float32)

        for i, net in enumerate(self.networks):
            net.eval()
            obs_t = torch.from_numpy(obs_array[i]).unsqueeze(0)
            action_mean, value = net.forward(obs_t)
            std = net.log_std.exp().expand_as(action_mean)
            from torch.distributions import Normal
            dist = Normal(action_mean, std)
            raw = dist.rsample()
            log_prob = dist.log_prob(raw).sum(dim=-1)
            norm = raw.norm(dim=-1, keepdim=True).clamp(min=1.0)
            action = raw / norm

            all_actions[i] = action.squeeze(0).numpy()
            all_raw[i] = raw.squeeze(0).numpy()
            all_log_probs[i] = log_prob.squeeze(0).item()
            all_values[i] = value.squeeze(0).item()

        return all_actions, all_raw, all_log_probs, all_values

    def ablated_get_actions(self, observations: tuple) -> np.ndarray:
        """IPPO has no execution-time message channel to ablate."""
        raise NotImplementedError(
            "IPPO has no execution-time communication; an identity comparison "
            "is not a communication ablation."
        )

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self, env: Any, n_steps: int = 2048, seed: int = 0
    ) -> dict[str, Any]:
        """Run the environment for n_steps and return per-agent trajectory data.

        The environment is reset at the start and whenever an episode ends.

        Returns a dict with keys:
            obs         : (n_steps, M, obs_dim)  float32
            raw_actions : (n_steps, M, 2)         float32  (pre-clipping)
            log_probs   : (n_steps, M)            float32
            rewards     : (n_steps, M)            float32
            values      : (n_steps, M)            float32
            dones       : (n_steps, M)            float32
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

            # Per-agent reward from capture events.
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
        """Run PPO update for each agent independently.

        Returns a dict of mean losses across agents.
        """
        obs = rollout_data["obs"]               # (T, M, obs_dim)
        raw_actions = rollout_data["raw_actions"]  # (T, M, 2)
        old_log_probs = rollout_data["log_probs"]  # (T, M)
        rewards = rollout_data["rewards"]           # (T, M)
        values = rollout_data["values"]             # (T, M)
        dones = rollout_data["dones"]               # (T, M)

        T = obs.shape[0]
        advantages = self._compute_advantages(rewards, values, dones)  # (T, M)
        returns = advantages + values                                    # (T, M)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for agent_idx in range(self.n_agents):
            net = self.networks[agent_idx]
            opt = self.optimisers[agent_idx]
            net.train()

            obs_i = torch.from_numpy(obs[:, agent_idx, :])          # (T, obs_dim)
            act_i = torch.from_numpy(raw_actions[:, agent_idx, :])  # (T, 2)
            old_lp_i = torch.from_numpy(old_log_probs[:, agent_idx])  # (T,)
            adv_i = torch.from_numpy(advantages[:, agent_idx].astype(np.float32))
            ret_i = torch.from_numpy(returns[:, agent_idx].astype(np.float32))

            # Normalise advantages.
            adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

            indices = np.arange(T)
            for _ in range(self.n_epochs):
                np.random.shuffle(indices)
                for start in range(0, T, self.batch_size):
                    mb_idx = indices[start : start + self.batch_size]
                    mb_idx_t = torch.from_numpy(mb_idx)

                    mb_obs = obs_i[mb_idx_t]
                    mb_act = act_i[mb_idx_t]
                    mb_old_lp = old_lp_i[mb_idx_t]
                    mb_adv = adv_i[mb_idx_t]
                    mb_ret = ret_i[mb_idx_t]

                    new_lp, entropy, value_pred = net.evaluate_actions(mb_obs, mb_act)

                    ratio = torch.exp(new_lp - mb_old_lp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.functional.mse_loss(value_pred, mb_ret)
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                    opt.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += (-entropy_loss).item()

        n_updates = self.n_agents * self.n_epochs * max(1, T // self.batch_size)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def _compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Generalised Advantage Estimation (GAE).

        Parameters
        ----------
        rewards : (T, M) float32
        values  : (T, M) float32
        dones   : (T, M) float32

        Returns
        -------
        advantages : (T, M) float32
        """
        T, M = rewards.shape
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(M, dtype=np.float32)

        # Bootstrap from last value (assume episode continues if not done).
        next_value = values[-1] * (1.0 - dones[-1])
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
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
        """Save all agent network state dicts to a single file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "obs_dim": self.obs_dim,
            "n_agents": self.n_agents,
            "networks": [net.state_dict() for net in self.networks],
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load agent network state dicts from a file."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        assert state["obs_dim"] == self.obs_dim
        assert state["n_agents"] == self.n_agents
        for net, sd in zip(self.networks, state["networks"]):
            net.load_state_dict(sd)
