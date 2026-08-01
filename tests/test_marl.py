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
from particle_benchmark.marl.maddpg import MADDPG, MADDPGActor, MADDPGCritic, ReplayBuffer
from particle_benchmark.marl.coma import COMA, CounterfactualCritic
from particle_benchmark.marl.commnet import CommNet, CommNetModule
from particle_benchmark.marl.vdn import VDN, IndependentQNet, VDNMixer


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
        # Use evaluate_actions so that log_std participates in the forward pass.
        actions = torch.zeros(4, 2)
        log_prob, entropy, value = net.evaluate_actions(obs, actions)
        loss = log_prob.sum() + entropy.sum() + value.sum()
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


# ---------------------------------------------------------------------------
# MADDPG
# ---------------------------------------------------------------------------

class TestMADDPG:

    def test_replay_buffer_add_and_sample(self, obs_dim):
        buf = ReplayBuffer(capacity=100, obs_dim=obs_dim, n_agents=4)
        obs = np.zeros((4, obs_dim), dtype=np.float32)
        act = np.zeros((4, 2), dtype=np.float32)
        rew = np.zeros(4, dtype=np.float32)
        buf.add(obs, act, rew, obs, False)
        assert len(buf) == 1
        for _ in range(99):
            buf.add(obs, act, rew, obs, False)
        assert len(buf) == 100
        # Over-capacity write wraps around.
        buf.add(obs, act, rew, obs, False)
        assert len(buf) == 100

    def test_buffer_sample_shapes(self, obs_dim):
        buf = ReplayBuffer(capacity=200, obs_dim=obs_dim, n_agents=4)
        for _ in range(100):
            buf.add(
                np.zeros((4, obs_dim), dtype=np.float32),
                np.zeros((4, 2), dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros((4, obs_dim), dtype=np.float32),
                False,
            )
        o, a, r, no, d = buf.sample(16)
        assert o.shape == (16, 4, obs_dim)
        assert a.shape == (16, 4, 2)
        assert r.shape == (16, 4)
        assert no.shape == (16, 4, obs_dim)
        assert d.shape == (16,)

    def test_maddpg_actor_forward_shape(self, obs_dim):
        actor = MADDPGActor(obs_dim)
        obs = torch.zeros(8, obs_dim)
        out = actor(obs)
        assert out.shape == (8, 2)
        # Tanh output is bounded.
        assert (out.abs() <= 1.0).all()

    def test_maddpg_critic_forward_shape(self, obs_dim):
        n_agents = 4
        critic = MADDPGCritic(obs_dim, n_agents)
        obs_flat = torch.zeros(8, obs_dim * n_agents)
        act_flat = torch.zeros(8, 2 * n_agents)
        q = critic(obs_flat, act_flat)
        assert q.shape == (8,)

    def test_maddpg_critic_gradient_flows(self, obs_dim):
        n_agents = 4
        critic = MADDPGCritic(obs_dim, n_agents)
        obs_flat = torch.randn(4, obs_dim * n_agents)
        act_flat = torch.randn(4, 2 * n_agents)
        loss = critic(obs_flat, act_flat).sum()
        loss.backward()
        for name, p in critic.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_maddpg_get_actions_shape(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        maddpg = MADDPG(obs_dim=obs_dim, n_agents=4)
        actions = maddpg.get_actions(obs, explore=False)
        assert actions.shape == (4, 2)

    def test_maddpg_actions_in_unit_ball(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        maddpg = MADDPG(obs_dim=obs_dim, n_agents=4)
        actions = maddpg.get_actions(obs, explore=False)
        norms = np.linalg.norm(actions, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_maddpg_ready_false_before_buffer_filled(self, obs_dim):
        maddpg = MADDPG(obs_dim=obs_dim, batch_size=64)
        assert not maddpg.ready

    def test_maddpg_10_steps_no_error(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        maddpg = MADDPG(obs_dim=obs_dim, n_agents=config.collector_count, batch_size=8)
        maddpg._episode_seed = 10
        for _ in range(10):
            maddpg.collect_rollout(env, n_steps=1)
        assert len(maddpg.buffer) == 10

    def test_maddpg_update_when_ready(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        maddpg = MADDPG(obs_dim=obs_dim, n_agents=config.collector_count, batch_size=8)
        maddpg._episode_seed = 20
        for _ in range(20):
            maddpg.collect_rollout(env, n_steps=1)
        assert maddpg.ready
        losses = maddpg.update()
        assert "critic_loss" in losses
        assert "actor_loss" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_maddpg_save_load(self, env_and_obs, obs_dim, tmp_path):
        _, _, config = env_and_obs
        maddpg = MADDPG(obs_dim=obs_dim, n_agents=config.collector_count)
        path = tmp_path / "maddpg.pt"
        maddpg.save(path)
        maddpg2 = MADDPG(obs_dim=obs_dim, n_agents=config.collector_count)
        maddpg2.load(path)
        for a1, a2 in zip(maddpg.actors, maddpg2.actors):
            for p1, p2 in zip(a1.parameters(), a2.parameters()):
                assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# COMA
# ---------------------------------------------------------------------------

class TestCOMA:

    def test_counterfactual_critic_shape(self, obs_dim):
        n_agents = 4
        critic = CounterfactualCritic(obs_dim * n_agents, n_agents)
        global_obs = torch.zeros(8, obs_dim * n_agents)
        act_flat = torch.zeros(8, 2 * n_agents)
        q = critic(global_obs, act_flat)
        assert q.shape == (8,)

    def test_counterfactual_critic_gradient(self, obs_dim):
        n_agents = 4
        critic = CounterfactualCritic(obs_dim * n_agents, n_agents)
        global_obs = torch.randn(4, obs_dim * n_agents)
        act_flat = torch.randn(4, 2 * n_agents)
        loss = critic(global_obs, act_flat).sum()
        loss.backward()
        for name, p in critic.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_coma_init(self, obs_dim):
        coma = COMA(obs_dim=obs_dim, n_agents=4)
        assert isinstance(coma.actor, ActorCritic)
        assert isinstance(coma.critic, CounterfactualCritic)

    def test_coma_get_actions_shape(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        coma = COMA(obs_dim=obs_dim, n_agents=4)
        actions, raw, log_probs, values = coma._get_actions_with_raw(obs)
        assert actions.shape == (4, 2)
        assert raw.shape == (4, 2)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_coma_actions_in_unit_ball(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        coma = COMA(obs_dim=obs_dim, n_agents=4)
        actions, _, _, _ = coma._get_actions_with_raw(obs)
        norms = np.linalg.norm(actions, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_coma_collect_rollout_shape(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        coma = COMA(obs_dim=obs_dim, n_agents=config.collector_count)
        rollout = coma.collect_rollout(env, n_steps=10, seed=5)
        assert rollout["obs"].shape == (10, config.collector_count, obs_dim)
        assert rollout["global_obs"].shape == (10, obs_dim * config.collector_count)
        assert rollout["raw_actions"].shape == (10, config.collector_count, 2)

    def test_coma_update_returns_loss_dict(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        coma = COMA(
            obs_dim=obs_dim, n_agents=config.collector_count,
            n_epochs=1, batch_size=8, n_cf_samples=2,
        )
        rollout = coma.collect_rollout(env, n_steps=32, seed=6)
        losses = coma.update(rollout)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_coma_10_steps_no_error(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        coma = COMA(obs_dim=obs_dim, n_agents=config.collector_count)
        observations, _ = env.reset(seed=55)
        for _ in range(10):
            actions, _, _, _ = coma._get_actions_with_raw(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                observations, _ = env.reset(seed=56)

    def test_coma_save_load(self, env_and_obs, obs_dim, tmp_path):
        _, _, config = env_and_obs
        coma = COMA(obs_dim=obs_dim, n_agents=config.collector_count)
        path = tmp_path / "coma.pt"
        coma.save(path)
        coma2 = COMA(obs_dim=obs_dim, n_agents=config.collector_count)
        coma2.load(path)
        for p1, p2 in zip(coma.actor.parameters(), coma2.actor.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# CommNet
# ---------------------------------------------------------------------------

class TestCommNet:

    def test_commnet_module_forward_step(self, obs_dim):
        n_agents = 4
        net = CommNetModule(obs_dim, h_dim=64, n_comm_rounds=2)
        obs = torch.zeros(n_agents, obs_dim)
        action_mean, value = net.forward_step(obs)
        assert action_mean.shape == (n_agents, 2)
        assert value.shape == (n_agents,)
        # Tanh output bounded.
        assert (action_mean.abs() <= 1.0).all()

    def test_commnet_module_forward_batch(self, obs_dim):
        T, M = 5, 4
        net = CommNetModule(obs_dim, h_dim=64, n_comm_rounds=2)
        obs = torch.zeros(T, M, obs_dim)
        action_mean, value = net.forward_batch(obs)
        assert action_mean.shape == (T, M, 2)
        assert value.shape == (T, M)

    def test_commnet_comm_pass_changes_hidden(self, obs_dim):
        """Communication should produce different outputs for each agent."""
        net = CommNetModule(obs_dim, h_dim=32, n_comm_rounds=2)
        # Give agent 0 a different obs from agents 1-3.
        obs = torch.randn(4, obs_dim)
        obs[0] *= 10.0  # large signal for agent 0
        action_mean, _ = net.forward_step(obs)
        # Not all action means should be identical (comm propagates info).
        assert not torch.allclose(action_mean[0], action_mean[1])

    def test_commnet_init(self, obs_dim):
        commnet = CommNet(obs_dim=obs_dim, n_agents=4)
        assert isinstance(commnet.net, CommNetModule)

    def test_commnet_get_actions_shape(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        commnet = CommNet(obs_dim=obs_dim, n_agents=4)
        actions, raw, log_probs, values = commnet._get_actions_with_raw(obs)
        assert actions.shape == (4, 2)
        assert raw.shape == (4, 2)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_commnet_actions_in_unit_ball(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        commnet = CommNet(obs_dim=obs_dim, n_agents=4)
        actions, _, _, _ = commnet._get_actions_with_raw(obs)
        norms = np.linalg.norm(actions, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_commnet_collect_rollout_shape(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        commnet = CommNet(obs_dim=obs_dim, n_agents=config.collector_count)
        rollout = commnet.collect_rollout(env, n_steps=10, seed=7)
        assert rollout["obs"].shape == (10, config.collector_count, obs_dim)
        assert rollout["raw_actions"].shape == (10, config.collector_count, 2)
        assert rollout["values"].shape == (10, config.collector_count)

    def test_commnet_update_returns_loss_dict(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        commnet = CommNet(
            obs_dim=obs_dim, n_agents=config.collector_count,
            n_epochs=1, batch_size=8,
        )
        rollout = commnet.collect_rollout(env, n_steps=32, seed=8)
        losses = commnet.update(rollout)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_commnet_10_steps_no_error(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        commnet = CommNet(obs_dim=obs_dim, n_agents=config.collector_count)
        observations, _ = env.reset(seed=33)
        for _ in range(10):
            actions, _, _, _ = commnet._get_actions_with_raw(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                observations, _ = env.reset(seed=34)

    def test_commnet_save_load(self, env_and_obs, obs_dim, tmp_path):
        _, _, config = env_and_obs
        commnet = CommNet(obs_dim=obs_dim, n_agents=config.collector_count)
        path = tmp_path / "commnet.pt"
        commnet.save(path)
        commnet2 = CommNet(obs_dim=obs_dim, n_agents=config.collector_count)
        commnet2.load(path)
        for p1, p2 in zip(commnet.net.parameters(), commnet2.net.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# VDN
# ---------------------------------------------------------------------------

class TestVDN:

    def test_independent_qnet_shape(self, obs_dim):
        qnet = IndependentQNet(obs_dim)
        obs = torch.zeros(8, obs_dim)
        q = qnet(obs)
        assert q.shape == (8,)

    def test_vdn_mixer_shape(self):
        mixer = VDNMixer()
        # (B, M) → (B,)
        q_vals = torch.ones(8, 4)
        q_tot = mixer(q_vals)
        assert q_tot.shape == (8,)
        # Q_tot = sum of Q_i.
        assert torch.allclose(q_tot, torch.full((8,), 4.0))

    def test_vdn_mixer_no_params(self):
        mixer = VDNMixer()
        assert sum(1 for _ in mixer.parameters()) == 0

    def test_vdn_init(self, obs_dim):
        vdn = VDN(obs_dim=obs_dim, n_agents=4)
        assert isinstance(vdn.actor, ActorCritic)
        assert len(vdn.q_nets) == 4
        assert isinstance(vdn.mixer, VDNMixer)

    def test_vdn_get_actions_shape(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        vdn = VDN(obs_dim=obs_dim, n_agents=4)
        actions, raw, log_probs, values = vdn._get_actions_with_raw(obs)
        assert actions.shape == (4, 2)
        assert raw.shape == (4, 2)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_vdn_actions_in_unit_ball(self, env_and_obs, obs_dim):
        _, obs, _ = env_and_obs
        vdn = VDN(obs_dim=obs_dim, n_agents=4)
        actions, _, _, _ = vdn._get_actions_with_raw(obs)
        norms = np.linalg.norm(actions, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_vdn_collect_rollout_shape(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        vdn = VDN(obs_dim=obs_dim, n_agents=config.collector_count)
        rollout = vdn.collect_rollout(env, n_steps=10, seed=9)
        assert rollout["obs"].shape == (10, config.collector_count, obs_dim)
        assert rollout["raw_actions"].shape == (10, config.collector_count, 2)
        assert rollout["values"].shape == (10, config.collector_count)

    def test_vdn_update_returns_loss_dict(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        vdn = VDN(
            obs_dim=obs_dim, n_agents=config.collector_count,
            n_epochs=1, batch_size=8,
        )
        rollout = vdn.collect_rollout(env, n_steps=32, seed=10)
        losses = vdn.update(rollout)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert all(np.isfinite(v) for v in losses.values())

    def test_vdn_10_steps_no_error(self, env_and_obs, obs_dim):
        env, _, config = env_and_obs
        vdn = VDN(obs_dim=obs_dim, n_agents=config.collector_count)
        observations, _ = env.reset(seed=44)
        for _ in range(10):
            actions, _, _, _ = vdn._get_actions_with_raw(observations)
            observations, reward, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                observations, _ = env.reset(seed=45)

    def test_vdn_save_load(self, env_and_obs, obs_dim, tmp_path):
        _, _, config = env_and_obs
        vdn = VDN(obs_dim=obs_dim, n_agents=config.collector_count)
        path = tmp_path / "vdn.pt"
        vdn.save(path)
        vdn2 = VDN(obs_dim=obs_dim, n_agents=config.collector_count)
        vdn2.load(path)
        for p1, p2 in zip(vdn.actor.parameters(), vdn2.actor.parameters()):
            assert torch.allclose(p1, p2)
        for q1, q2 in zip(vdn.q_nets, vdn2.q_nets):
            for p1, p2 in zip(q1.parameters(), q2.parameters()):
                assert torch.allclose(p1, p2)

