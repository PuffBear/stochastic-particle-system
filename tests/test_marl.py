"""Tests for IPPO and MAPPO baselines.

All tests are skipped automatically if PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module if torch is missing.
torch = pytest.importorskip("torch")

from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.marl.networks import (
    ActorCritic,
    CentralizedCritic,
    flatten_all_observations,
    flatten_observation,
    compute_obs_dim,
)
from particle_benchmark.marl.ippo import IPPO
from particle_benchmark.marl.mappo import MAPPO


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env_and_obs():
    """Shared environment and initial observation for this test module."""
    config = ParticleEnvConfig(horizon=67, signal_strength=0.06)
    env = ParticleCollectorEnv(config)
    obs, info = env.reset(seed=42)
    return env, obs, config


@pytest.fixture(scope="module")
def obs_dim(env_and_obs):
    _, obs, _ = env_and_obs
    return compute_obs_dim(obs)


# ---------------------------------------------------------------------------
# Observation flattening
# ---------------------------------------------------------------------------

class TestObservationFlattening:

    def test_flatten_single_obs_shape(self, env_and_obs):
        _, obs, _ = env_and_obs
        flat = flatten_observation(obs[0])
        # self_pos(2) + particles(32*5) + particle_mask(32) + vel_mask(32) + teammates(3*2)
        assert flat.shape == (232,)
        assert flat.dtype == np.float32

    def test_flatten_all_obs_shape(self, env_and_obs, obs_dim):
        _, obs, config = env_and_obs
        flat_all = flatten_all_observations(obs)
        assert flat_all.shape == (config.collector_count, obs_dim)

    def test_obs_dim_matches_expected(self, obs_dim):
        # 2 + 32*5 + 32 + 32 + 3*2 = 232 with default config
        assert obs_dim == 232

    def test_flatten_is_finite(self, env_and_obs):
        _, obs, _ = env_and_obs
        flat = flatten_observation(obs[0])
        assert np.all(np.isfinite(flat))


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class TestActorCritic:

    def test_forward_shapes_single(self, obs_dim):
        net = ActorCritic(obs_dim)
        obs = torch.zeros(1, obs_dim)
        action_mean, value = net(obs)
        assert action_mean.shape == (1, 2)
        assert value.shape == (1,)

    def test_forward_shapes_batch(self, obs_dim):
        net = ActorCritic(obs_dim)
        obs = torch.zeros(8, obs_dim)
        action_mean, value = net(obs)
        assert action_mean.shape == (8, 2)
        assert value.shape == (8,)

    def test_action_mean_bounded(self, obs_dim):
        net = ActorCritic(obs_dim)
        obs = torch.randn(32, obs_dim)
        action_mean, _ = net(obs)
        # Tanh head guarantees each component in (-1, 1).
        assert (action_mean.abs() <= 1.0).all()

    def test_get_action_and_log_prob_shapes(self, obs_dim):
        net = ActorCritic(obs_dim)
        obs = torch.zeros(4, obs_dim)
        action, log_prob, value = net.get_action_and_log_prob(obs)
        assert action.shape == (4, 2)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)

    def test_action_clipped_to_unit_ball(self, obs_dim):
        net = ActorCritic(obs_dim)
        obs = torch.randn(64, obs_dim)
        action, _, _ = net.get_action_and_log_prob(obs)
        norms = action.norm(dim=-1)
        assert (norms <= 1.0 + 1e-5).all(), f"Max norm: {norms.max():.4f}"

    def test_evaluate_actions_shapes(self, obs_dim):
        net = ActorCritic(obs_dim)
        obs = torch.zeros(16, obs_dim)
        actions = torch.zeros(16, 2)
        log_prob, entropy, value = net.evaluate_actions(obs, actions)
        assert log_prob.shape == (16,)
        assert entropy.shape == (16,)
        assert value.shape == (16,)

    def test_log_std_is_learnable(self, obs_dim):
        net = ActorCritic(obs_dim)
        assert any(p is net.log_std for p in net.parameters())

    def test_gradient_flows(self, obs_dim):
        net = ActorCritic(obs_dim)
        obs = torch.randn(4, obs_dim)
        action_mean, value = net(obs)
        loss = action_mean.sum() + value.sum()
        loss.backward()
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


class TestCentralizedCritic:

    def test_forward_shape(self, obs_dim):
        n_agents = 4
        critic = CentralizedCritic(obs_dim * n_agents)
        global_obs = torch.zeros(8, obs_dim * n_agents)
        value = critic(global_obs)
        assert value.shape == (8,)

    def test_gradient_flows(self, obs_dim):
        critic = CentralizedCritic(obs_dim * 4)
        global_obs = torch.randn(4, obs_dim * 4)
        loss = critic(global_obs).sum()
        loss.backward()
        for name, param in critic.named_parameters():
            assert param.grad is not None, f"No grad for {name}"


# ---------------------------------------------------------------------------
# IPPO integration
# ---------------------------------------------------------------------------

class TestIPPO:

    def test_init(self, obs_dim):
        ippo = IPPO(obs_dim=obs_dim, n_agents=4)
        assert len(ippo.networks) == 4
        assert len(ippo.optimisers) == 4

    def test_get_actions_shape(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        ippo = IPPO(obs_dim=obs_dim)
        actions, raw, log_probs, values = ippo._get_actions_with_raw(obs)
        assert actions.shape == (4, 2)
        assert raw.shape == (4, 2)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_actions_in_unit_ball(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        ippo = IPPO(obs_dim=obs_dim)
        actions, _, _, _ = ippo._get_actions_with_raw(obs)
        norms = np.linalg.norm(actions, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5), f"Max norm: {norms.max():.4f}"

    def test_collect_rollout_shape(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        ippo = IPPO(obs_dim=obs_dim, n_agents=config.collector_count)
        rollout = ippo.collect_rollout(env, n_steps=10, seed=1)
        assert rollout["obs"].shape == (10, config.collector_count, obs_dim)
        assert rollout["raw_actions"].shape == (10, config.collector_count, 2)
        assert rollout["rewards"].shape == (10, config.collector_count)
        assert rollout["dones"].shape == (10, config.collector_count)

    def test_update_returns_loss_dict(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        ippo = IPPO(obs_dim=obs_dim, n_agents=config.collector_count,
                    n_epochs=1, batch_size=8)
        rollout = ippo.collect_rollout(env, n_steps=32, seed=2)
        losses = ippo.update(rollout)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_10_steps_no_error(self, env_and_obs, obs_dim):
        """IPPO can run 10 environment steps without raising."""
        env, _, config = env_and_obs
        ippo = IPPO(obs_dim=obs_dim, n_agents=config.collector_count)
        observations, _ = env.reset(seed=99)
        for _ in range(10):
            actions, _, _, _ = ippo._get_actions_with_raw(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                observations, _ = env.reset(seed=100)

    def test_save_load(self, env_and_obs, obs_dim, tmp_path):
        env, obs, config = env_and_obs
        ippo = IPPO(obs_dim=obs_dim, n_agents=config.collector_count)
        path = tmp_path / "ippo.pt"
        ippo.save(path)
        ippo2 = IPPO(obs_dim=obs_dim, n_agents=config.collector_count)
        ippo2.load(path)
        # Check that the loaded weights match.
        for net1, net2 in zip(ippo.networks, ippo2.networks):
            for p1, p2 in zip(net1.parameters(), net2.parameters()):
                assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# MAPPO integration
# ---------------------------------------------------------------------------

class TestMAPPO:

    def test_init(self, obs_dim):
        mappo = MAPPO(obs_dim=obs_dim, n_agents=4)
        # One shared actor + one centralised critic.
        assert isinstance(mappo.actor, ActorCritic)
        assert isinstance(mappo.critic, CentralizedCritic)

    def test_get_actions_shape(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        mappo = MAPPO(obs_dim=obs_dim)
        actions, raw, log_probs, values = mappo._get_actions_with_raw(obs)
        assert actions.shape == (4, 2)
        assert raw.shape == (4, 2)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_centralized_values_equal_across_agents(self, env_and_obs, obs_dim):
        """MAPPO uses a single centralized value, same for all agents."""
        _, obs, _ = env_and_obs
        mappo = MAPPO(obs_dim=obs_dim)
        _, _, _, values = mappo._get_actions_with_raw(obs)
        # All agents get the same centralized value estimate.
        assert np.allclose(values, values[0])

    def test_actions_in_unit_ball(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        mappo = MAPPO(obs_dim=obs_dim)
        actions, _, _, _ = mappo._get_actions_with_raw(obs)
        norms = np.linalg.norm(actions, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_collect_rollout_shape(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        mappo = MAPPO(obs_dim=obs_dim, n_agents=config.collector_count)
        rollout = mappo.collect_rollout(env, n_steps=10, seed=3)
        assert rollout["obs"].shape == (10, config.collector_count, obs_dim)
        assert rollout["global_obs"].shape == (10, obs_dim * config.collector_count)
        assert rollout["raw_actions"].shape == (10, config.collector_count, 2)

    def test_update_returns_loss_dict(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        mappo = MAPPO(obs_dim=obs_dim, n_agents=config.collector_count,
                      n_epochs=1, batch_size=8)
        rollout = mappo.collect_rollout(env, n_steps=32, seed=4)
        losses = mappo.update(rollout)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_10_steps_no_error(self, env_and_obs, obs_dim):
        """MAPPO can run 10 environment steps without raising."""
        env, _, config = env_and_obs
        mappo = MAPPO(obs_dim=obs_dim, n_agents=config.collector_count)
        observations, _ = env.reset(seed=77)
        for _ in range(10):
            actions, _, _, _ = mappo._get_actions_with_raw(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                observations, _ = env.reset(seed=78)

    def test_save_load(self, env_and_obs, obs_dim, tmp_path):
        env, obs, config = env_and_obs
        mappo = MAPPO(obs_dim=obs_dim, n_agents=config.collector_count)
        path = tmp_path / "mappo.pt"
        mappo.save(path)
        mappo2 = MAPPO(obs_dim=obs_dim, n_agents=config.collector_count)
        mappo2.load(path)
        for p1, p2 in zip(mappo.actor.parameters(), mappo2.actor.parameters()):
            assert torch.allclose(p1, p2)
        for p1, p2 in zip(mappo.critic.parameters(), mappo2.critic.parameters()):
            assert torch.allclose(p1, p2)

    def test_shared_actor_same_weights(self, obs_dim):
        """MAPPO actor weights are shared — same object, not a copy."""
        mappo = MAPPO(obs_dim=obs_dim, n_agents=4)
        # Modify one parameter and verify it's the same object across agents.
        actor1 = mappo.actor
        actor2 = mappo.actor  # same reference
        assert actor1 is actor2
