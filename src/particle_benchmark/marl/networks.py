"""Neural network architectures for IPPO and MAPPO baselines.

Requires PyTorch: pip install 'stochastic-particle-system[marl]'

Observation flattening contract
--------------------------------
Each LocalObservation dict is flattened to a 1-D float32 vector in this order:

    self_position                      (2,)
    particles.flatten()                (nearest_particles_k * 5,)   default 32*5=160
    particle_mask.astype(float32)      (nearest_particles_k,)        default 32
    velocity_valid_mask.astype(float32)(nearest_particles_k,)        default 32
    teammate_relative_positions.flatten()  (n_teammates * 2,)        default 3*2=6

With default ParticleEnvConfig this gives obs_dim = 2 + 160 + 32 + 32 + 6 = 232.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    raise ImportError(
        "MARL baselines require PyTorch: pip install 'stochastic-particle-system[marl]'"
    )


# ---------------------------------------------------------------------------
# Observation utilities
# ---------------------------------------------------------------------------

def flatten_observation(obs: dict) -> NDArray[np.float32]:
    """Flatten a LocalObservation dict into a 1-D float32 numpy array."""
    parts: list[NDArray[np.float32]] = [
        obs["self_position"].astype(np.float32),
        obs["particles"].flatten().astype(np.float32),
        obs["particle_mask"].astype(np.float32),
        obs["velocity_valid_mask"].astype(np.float32),
    ]
    if "teammate_relative_positions" in obs:
        parts.append(obs["teammate_relative_positions"].flatten().astype(np.float32))
    return np.concatenate(parts)


def flatten_all_observations(observations: tuple) -> NDArray[np.float32]:
    """Flatten a tuple of LocalObservation dicts; returns (M, obs_dim) array."""
    return np.stack([flatten_observation(obs) for obs in observations], axis=0)


def compute_obs_dim(observations: tuple) -> int:
    """Return the flat observation dimension from one set of per-agent observations."""
    return flatten_observation(observations[0]).shape[0]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared-backbone actor-critic for one agent.

    Architecture:
        backbone:  Linear(obs_dim, 128) -> ReLU -> Linear(128, 64) -> ReLU
        actor:     Linear(64, 2) -> Tanh   [action mean in (-1,1)^2]
        critic:    Linear(64, 1)           [state-value estimate]
        log_std:   learnable scalar per action dimension, initialised to 0

    ``forward`` returns (action_mean, value).
    ``get_action_and_log_prob`` samples from Normal(action_mean, std), clips
    the sample to the unit ball, and returns (clipped_action, log_prob, value).
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(nn.Linear(64, 2), nn.Tanh())
        self.critic_head = nn.Linear(64, 1)
        # Learnable log-std (one value per action dimension, shared across agents).
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_mean, value).  obs: (..., obs_dim)."""
        h = self.backbone(obs)
        action_mean = self.actor_head(h)
        value = self.critic_head(h).squeeze(-1)
        return action_mean, value

    def get_action_and_log_prob(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from Normal distribution and clip to unit ball.

        Returns
        -------
        action : Tensor (..., 2)  clipped to unit ball
        log_prob : Tensor (...)   log-prob of the *raw* (pre-clipping) sample,
                                   summed over action dimensions
        value : Tensor (...)
        """
        action_mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)
        raw_action = dist.rsample()
        # Log-prob of the raw sample (before clipping), summed over action dims.
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        # Clip to unit ball: divide by norm when norm > 1.
        norm = raw_action.norm(dim=-1, keepdim=True).clamp(min=1.0)
        action = raw_action / norm
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored (raw) actions under the current policy.

        Parameters
        ----------
        obs     : Tensor (B, obs_dim)
        actions : Tensor (B, 2)   *raw* pre-clipping actions stored at rollout time

        Returns
        -------
        log_prob  : Tensor (B,)
        entropy   : Tensor (B,)
        value     : Tensor (B,)
        """
        action_mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


# ---------------------------------------------------------------------------
# Centralised critic (MAPPO)
# ---------------------------------------------------------------------------

class CentralizedCritic(nn.Module):
    """MAPPO centralised critic that consumes all agents' observations.

    Input dimension: obs_dim * n_agents  (concatenated flat observations).
    """

    def __init__(self, global_obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """Return value estimate.  global_obs: (..., obs_dim * n_agents)."""
        return self.net(global_obs).squeeze(-1)
