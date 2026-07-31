"""Multi-Agent Deep Deterministic Policy Gradient (MADDPG) for the particle-collector benchmark.

Reference: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
Environments", NeurIPS 2017.

Off-policy: each agent has a deterministic Actor and a centralised Critic that
observes all agents' observations and actions.  Exploration uses Gaussian noise;
target networks are updated with Polyak averaging.

Requires PyTorch: pip install 'stochastic-particle-system[marl]'
"""

from __future__ import annotations

import copy
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

from .networks import flatten_all_observations


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity circular replay buffer for MADDPG transitions.

    Each transition stores full-team tensors:
        obs_all      : (M, obs_dim)
        actions_all  : (M, 2)
        rewards_all  : (M,)
        next_obs_all : (M, obs_dim)
        done         : scalar float
    """

    def __init__(self, capacity: int, obs_dim: int, n_agents: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self._pos = 0
        self._size = 0

        self._obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, n_agents, 2), dtype=np.float32)
        self._rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self._next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs_all: np.ndarray,
        actions_all: np.ndarray,
        rewards_all: np.ndarray,
        next_obs_all: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition.  Arrays are (M, …) shaped."""
        self._obs[self._pos] = obs_all
        self._actions[self._pos] = actions_all
        self._rewards[self._pos] = rewards_all
        self._next_obs[self._pos] = next_obs_all
        self._dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Return a random mini-batch of (obs, actions, rewards, next_obs, dones)."""
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class MADDPGActor(nn.Module):
    """Deterministic actor for one agent.

    Architecture: Linear(obs_dim, 128) → ReLU → Linear(128, 64) → ReLU
                  → Linear(64, 2) → Tanh
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action in (-1,1)^2.  obs: (..., obs_dim)."""
        return self.net(obs)


class MADDPGCritic(nn.Module):
    """Centralised critic for one agent.

    Receives all agents' flattened observations and all agents' actions.
    Input dimension: obs_dim * n_agents + 2 * n_agents.
    """

    def __init__(self, obs_dim: int, n_agents: int) -> None:
        super().__init__()
        input_dim = obs_dim * n_agents + 2 * n_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, obs_flat: torch.Tensor, act_flat: torch.Tensor
    ) -> torch.Tensor:
        """Return Q-value estimate.

        Parameters
        ----------
        obs_flat : Tensor (..., M * obs_dim)  concatenated observations
        act_flat : Tensor (..., M * 2)        concatenated actions
        """
        x = torch.cat([obs_flat, act_flat], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MADDPG
# ---------------------------------------------------------------------------

class MADDPG:
    """Multi-Agent Deep Deterministic Policy Gradient.

    Parameters
    ----------
    obs_dim     : flat per-agent observation dimension (232 with default config)
    n_agents    : number of collectors (default 4)
    lr_actor    : Adam learning rate for actors
    lr_critic   : Adam learning rate for critics
    gamma       : discount factor
    tau         : Polyak averaging coefficient for target networks
    batch_size  : mini-batch size for updates
    buffer_size : replay buffer capacity
    noise_std   : standard deviation of Gaussian exploration noise
    """

    def __init__(
        self,
        obs_dim: int,
        n_agents: int = 4,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        noise_std: float = 0.1,
    ) -> None:
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std

        # One actor and one critic per agent.
        self.actors = [MADDPGActor(obs_dim) for _ in range(n_agents)]
        self.critics = [MADDPGCritic(obs_dim, n_agents) for _ in range(n_agents)]

        # Target networks (hard-copy initialisation).
        self.target_actors = [copy.deepcopy(a) for a in self.actors]
        self.target_critics = [copy.deepcopy(c) for c in self.critics]
        for net in self.target_actors + self.target_critics:
            for p in net.parameters():
                p.requires_grad_(False)

        # Optimisers.
        self.actor_optims = [
            optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors
        ]
        self.critic_optims = [
            optim.Adam(c.parameters(), lr=lr_critic) for c in self.critics
        ]

        # Replay buffer.
        self.buffer = ReplayBuffer(buffer_size, obs_dim, n_agents)

        # Persistent environment state for off-policy collection.
        self._obs: Any = None
        self._episode_seed: int = 0

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once the buffer has enough samples to form a mini-batch."""
        return len(self.buffer) >= self.batch_size

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_actions(self, observations: tuple, explore: bool = True) -> np.ndarray:
        """Return actions for all agents, optionally with exploration noise.

        Parameters
        ----------
        observations : tuple of M LocalObservation dicts
        explore      : add Gaussian noise if True (training); deterministic if False

        Returns
        -------
        actions : (M, 2) numpy array, clipped to the unit ball
        """
        flat_obs = flatten_all_observations(observations)  # (M, obs_dim)
        actions = np.zeros((self.n_agents, 2), dtype=np.float32)

        for i, actor in enumerate(self.actors):
            actor.eval()
            obs_t = torch.from_numpy(flat_obs[i]).unsqueeze(0)  # (1, obs_dim)
            action = actor(obs_t).squeeze(0).numpy()             # (2,)

            if explore:
                action = action + np.random.randn(2).astype(np.float32) * self.noise_std

            # Clip to unit ball.
            norm = np.linalg.norm(action)
            if norm > 1.0:
                action = action / norm
            actions[i] = action

        return actions

    def ablated_get_actions(self, observations: tuple) -> np.ndarray:
        """Return deterministic actions (M, 2) — communication ablation for MADDPG.

        MADDPG's centralised critics are used only during training; at inference
        actors are already independent (each actor observes only its own state).
        The ablation of communication therefore means disabling exploration noise
        so actors act deterministically, which corresponds to the standard
        deterministic evaluation mode. This is equivalent to
        get_actions(explore=False). It is provided to satisfy the common
        ablation interface for Coordination Efficiency computation.
        """
        return self.get_actions(observations, explore=False)

    # ------------------------------------------------------------------
    # Rollout collection (off-policy)
    # ------------------------------------------------------------------

    def collect_rollout(
        self, env: Any, n_steps: int = 1
    ) -> dict[str, Any]:
        """Step the environment n_steps times and store transitions in the buffer.

        Environment state is maintained across successive calls (off-policy style).
        The environment is auto-reset at episode boundaries.

        Returns a summary dict with keys:
            n_steps_collected : int
            n_episodes_done   : int
            total_reward      : float
            buffer_size       : int
        """
        if self._obs is None:
            self._obs, _ = env.reset(seed=self._episode_seed)

        n_done = 0
        total_reward = 0.0

        for _ in range(n_steps):
            flat_obs = flatten_all_observations(self._obs)  # (M, obs_dim)
            actions = self.get_actions(self._obs, explore=True)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            per_agent_reward = np.array(reward, dtype=np.float32)
            total_reward += float(per_agent_reward.sum())

            done = terminated or truncated
            flat_next_obs = flatten_all_observations(next_obs)

            self.buffer.add(flat_obs, actions, per_agent_reward, flat_next_obs, done)

            if done:
                self._episode_seed += 1
                self._obs, _ = env.reset(seed=self._episode_seed)
                n_done += 1
            else:
                self._obs = next_obs

        return {
            "n_steps_collected": n_steps,
            "n_episodes_done": n_done,
            "total_reward": total_reward,
            "buffer_size": len(self.buffer),
        }

    # ------------------------------------------------------------------
    # MADDPG update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        """Sample a mini-batch and update critics then actors.

        Returns
        -------
        dict with keys 'critic_loss' and 'actor_loss' (means across agents).
        Empty dict if buffer is not yet ready.
        """
        if not self.ready:
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        # shapes: (B, M, obs_dim), (B, M, 2), (B, M), (B, M, obs_dim), (B,)

        B = obs.shape[0]

        obs_t = torch.from_numpy(obs)         # (B, M, obs_dim)
        act_t = torch.from_numpy(actions)     # (B, M, 2)
        rew_t = torch.from_numpy(rewards)     # (B, M)
        nobs_t = torch.from_numpy(next_obs)   # (B, M, obs_dim)
        done_t = torch.from_numpy(dones)      # (B,)

        obs_flat = obs_t.reshape(B, -1)   # (B, M*obs_dim)
        act_flat = act_t.reshape(B, -1)   # (B, M*2)
        nobs_flat = nobs_t.reshape(B, -1) # (B, M*obs_dim)

        # Target actions for the next state (no gradient).
        with torch.no_grad():
            next_act_list = [
                self.target_actors[i](nobs_t[:, i, :])
                for i in range(self.n_agents)
            ]
            next_act_flat = torch.stack(next_act_list, dim=1).reshape(B, -1)  # (B, M*2)

        total_critic_loss = 0.0
        total_actor_loss = 0.0

        for i in range(self.n_agents):
            # ---- Critic update ----
            with torch.no_grad():
                q_next = self.target_critics[i](nobs_flat, next_act_flat)  # (B,)
                q_target = rew_t[:, i] + self.gamma * (1.0 - done_t) * q_next

            q_pred = self.critics[i](obs_flat, act_flat)  # (B,)
            critic_loss = nn.functional.mse_loss(q_pred, q_target)

            self.critic_optims[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), max_norm=0.5)
            self.critic_optims[i].step()
            total_critic_loss += critic_loss.item()

            # ---- Actor update ----
            # Build action tensor where agent i's action is differentiable.
            current_acts = [act_t[:, j, :].detach() for j in range(self.n_agents)]
            current_acts[i] = self.actors[i](obs_t[:, i, :])  # differentiable
            current_act_flat = torch.stack(current_acts, dim=1).reshape(B, -1)

            actor_loss = -self.critics[i](obs_flat, current_act_flat).mean()

            self.actor_optims[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), max_norm=0.5)
            self.actor_optims[i].step()
            total_actor_loss += actor_loss.item()

        self._polyak_update()

        return {
            "critic_loss": total_critic_loss / self.n_agents,
            "actor_loss": total_actor_loss / self.n_agents,
        }

    def _polyak_update(self) -> None:
        """Soft-update target networks: θ_target ← τ·θ + (1-τ)·θ_target."""
        for target, source in zip(self.target_actors, self.actors):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)
        for target, source in zip(self.target_critics, self.critics):
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save all actor and critic state dicts to a single file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "obs_dim": self.obs_dim,
            "n_agents": self.n_agents,
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "target_actors": [a.state_dict() for a in self.target_actors],
            "target_critics": [c.state_dict() for c in self.target_critics],
        }
        torch.save(state, path)

    def load(self, path: str | Path) -> None:
        """Load actor and critic state dicts from a file."""
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        assert state["obs_dim"] == self.obs_dim
        assert state["n_agents"] == self.n_agents
        for a, sd in zip(self.actors, state["actors"]):
            a.load_state_dict(sd)
        for c, sd in zip(self.critics, state["critics"]):
            c.load_state_dict(sd)
        for a, sd in zip(self.target_actors, state["target_actors"]):
            a.load_state_dict(sd)
        for c, sd in zip(self.target_critics, state["target_critics"]):
            c.load_state_dict(sd)
