from __future__ import annotations

import unittest

import numpy as np

from particle_benchmark.dynamics.collectors import (
    advance_collectors,
    bounded_velocity,
)
from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.initialization import (
    collector_lattice,
    sample_capture_free_initial_state,
)
from particle_benchmark.observations import build_local_observations
from particle_benchmark.policies import (
    privileged_field_policy,
    random_action_tensor,
    random_policy,
    stationary_policy,
)


def _assert_observations_equal(
    testcase: unittest.TestCase,
    left: tuple[dict[str, np.ndarray], ...],
    right: tuple[dict[str, np.ndarray], ...],
) -> None:
    testcase.assertEqual(len(left), len(right))
    for left_agent, right_agent in zip(left, right, strict=True):
        testcase.assertEqual(left_agent.keys(), right_agent.keys())
        for key in left_agent:
            np.testing.assert_array_equal(left_agent[key], right_agent[key])


class TestCollectorDynamics(unittest.TestCase):
    def test_velocity_is_projected_to_speed_ball(self) -> None:
        velocity = bounded_velocity([[3.0, 4.0], [0.3, 0.4]], max_speed=0.2)
        np.testing.assert_allclose(velocity[0], [0.12, 0.16])
        np.testing.assert_allclose(velocity[1], [0.06, 0.08])
        self.assertTrue(np.all(np.linalg.norm(velocity, axis=1) <= 0.2 + 1e-15))

    def test_collector_step_is_dt_and_boundary_bounded(self) -> None:
        position = np.array([[0.99, 0.5], [0.1, 0.1]])
        actual = advance_collectors(
            position,
            [[1.0, 0.0], [0.0, -10.0]],
            dt=0.2,
            max_speed=0.1,
            arena_size=(1.0, 1.0),
        )
        np.testing.assert_allclose(actual, [[0.99, 0.5], [0.1, 0.08]])

    def test_nonfinite_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bounded_velocity([[np.nan, 0.0]], max_speed=1.0)


class TestInitialization(unittest.TestCase):
    def test_canonical_collector_lattice_is_frozen(self) -> None:
        np.testing.assert_array_equal(
            collector_lattice(4, (1.0, 1.0)),
            [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]],
        )

    def test_seeded_reset_is_exact_and_capture_free(self) -> None:
        first = sample_capture_free_initial_state(
            np.random.default_rng(9),
            particle_count=100,
            collector_count=4,
            arena_size=(1.0, 1.0),
            exclusion_radius=0.04,
        )
        second = sample_capture_free_initial_state(
            np.random.default_rng(9),
            particle_count=100,
            collector_count=4,
            arena_size=(1.0, 1.0),
            exclusion_radius=0.04,
        )
        np.testing.assert_array_equal(first.particle_positions, second.particle_positions)
        np.testing.assert_array_equal(first.collector_positions, second.collector_positions)
        distances = np.linalg.norm(
            first.particle_positions[:, None, :]
            - first.collector_positions[None, :, :],
            axis=2,
        )
        self.assertTrue(np.all(distances > 0.04))

    def test_canonical_reset_has_no_capture_for_100_seeds(self) -> None:
        env = ParticleCollectorEnv(ParticleEnvConfig(horizon=1))
        for seed in range(100):
            _, info = env.reset(seed=seed)
            assert env.particle_positions is not None
            assert env.collector_positions is not None
            assert env.capture_state is not None
            distances = np.linalg.norm(
                env.particle_positions[:, None, :]
                - env.collector_positions[None, :, :],
                axis=2,
            )
            self.assertTrue(np.all(distances > 0.012), msg=f"seed={seed}")
            self.assertEqual(info["captured_total"], 0)
            np.testing.assert_array_equal(env.capture_state.owner, -1)

    def test_impossible_reset_stops_boundedly(self) -> None:
        with self.assertRaises(RuntimeError):
            sample_capture_free_initial_state(
                np.random.default_rng(1),
                particle_count=1,
                collector_count=1,
                arena_size=(1.0, 1.0),
                exclusion_radius=2.0,
                max_attempts_per_particle=5,
            )


class TestInformationBoundary(unittest.TestCase):
    def test_far_particle_cannot_change_local_observation(self) -> None:
        collector = np.array([[0.1, 0.1]])
        common = np.array([[0.15, 0.1], [0.9, 0.9]])
        changed_far = np.array([[0.15, 0.1], [0.7, 0.8]])
        kwargs = dict(
            free_particle_mask=[True, True],
            arena_size=(1.0, 1.0),
            sensing_radius=0.1,
            nearest_particles_k=2,
            include_particle_velocity=False,
            include_teammates=False,
        )
        left = build_local_observations(common, collector, **kwargs)
        right = build_local_observations(changed_far, collector, **kwargs)
        _assert_observations_equal(self, left, right)
        np.testing.assert_array_equal(left[0]["particle_mask"], [True, False])

    def test_field_strength_and_future_noise_are_not_observed(self) -> None:
        null = ParticleCollectorEnv(
            ParticleEnvConfig(
                particle_count=8,
                collector_count=2,
                horizon=2,
                field_family="uniform",
                signal_strength=0.0,
                nearest_particles_k=4,
            )
        )
        signal = ParticleCollectorEnv(
            ParticleEnvConfig(
                particle_count=8,
                collector_count=2,
                horizon=2,
                field_family="uniform",
                signal_strength=100.0,
                nearest_particles_k=4,
            )
        )
        null_observation, _ = null.reset(seed=71)
        signal_observation, _ = signal.reset(seed=71)
        _assert_observations_equal(self, null_observation, signal_observation)

        assert null._noise is not None
        null._noise[:] = 1_000.0
        after_future_noise_change = null._observations()
        _assert_observations_equal(self, null_observation, after_future_noise_change)
        for observation in null_observation:
            forbidden = {
                "field",
                "signal_strength",
                "field_orientation",
                "global_state",
                "future_noise",
            }
            self.assertTrue(forbidden.isdisjoint(observation))

    def test_field_orientation_is_drawn_once_and_pair_matched(self) -> None:
        common = dict(
            particle_count=4,
            collector_count=2,
            horizon=2,
            field_family="uniform",
            field_kwargs={},
            nearest_particles_k=2,
        )
        null = ParticleCollectorEnv(ParticleEnvConfig(**common, signal_strength=0.0))
        signal = ParticleCollectorEnv(ParticleEnvConfig(**common, signal_strength=0.4))
        null.reset(seed=811)
        signal.reset(seed=811)
        assert null._episode_field_kwargs is not None
        assert signal._episode_field_kwargs is not None
        self.assertEqual(
            null._episode_field_kwargs["orientation"],
            signal._episode_field_kwargs["orientation"],
        )
        orientation = null._episode_field_kwargs["orientation"]
        null._observations()
        self.assertEqual(null._episode_field_kwargs["orientation"], orientation)

    def test_teammate_relative_positions_are_normalized_in_fixed_id_order(self) -> None:
        observations = build_local_observations(
            [[0.5, 0.5]],
            [[0.1, 0.1], [0.4, 0.9]],
            [True],
            arena_size=(1.0, 1.0),
            sensing_radius=0.2,
            nearest_particles_k=1,
        )
        np.testing.assert_allclose(
            observations[0]["teammate_relative_positions"], [[0.3, 0.8]]
        )


class TestEnvironment(unittest.TestCase):
    def _small_config(self, **overrides: object) -> ParticleEnvConfig:
        values: dict[str, object] = {
            "particle_count": 12,
            "collector_count": 2,
            "horizon": 3,
            "nearest_particles_k": 4,
            "collector_radius": 0.01,
            "particle_radius": 0.0,
        }
        values.update(overrides)
        return ParticleEnvConfig(**values)

    def test_reset_shapes_and_no_initial_capture(self) -> None:
        env = ParticleCollectorEnv(self._small_config())
        observations, info = env.reset(seed=5)
        self.assertEqual(len(observations), 2)
        self.assertEqual(info["captured_total"], 0)
        assert env.capture_state is not None
        np.testing.assert_array_equal(env.capture_state.owner, -1)
        self.assertEqual(observations[0]["particles"].shape, (4, 5))
        self.assertFalse(np.any(observations[0]["velocity_valid_mask"]))
        self.assertEqual(env.config.particle_radius, 0.0)

    def test_rollout_is_exactly_reproducible(self) -> None:
        left = ParticleCollectorEnv(self._small_config())
        right = ParticleCollectorEnv(self._small_config())
        left.reset(seed=99)
        right.reset(seed=99)
        actions = np.array([[2.0, 0.0], [-1.0, 1.0]])
        left_step = left.step(actions)
        right_step = right.step(actions)
        _assert_observations_equal(self, left_step[0], right_step[0])
        np.testing.assert_array_equal(left_step[1], right_step[1])
        self.assertEqual(left_step[2:], right_step[2:])
        np.testing.assert_array_equal(left.particle_positions, right.particle_positions)
        np.testing.assert_array_equal(left.collector_positions, right.collector_positions)

    def test_capture_reward_ownership_and_termination(self) -> None:
        env = ParticleCollectorEnv(
            self._small_config(
                particle_count=1,
                diffusion_sigma=0.0,
                signal_strength=0.0,
            )
        )
        env.reset(seed=4)
        assert env.particle_positions is not None
        assert env.collector_positions is not None
        env.particle_positions[0] = env.collector_positions[1]
        _, reward, terminated, truncated, info = env.step(stationary_policy(2))
        np.testing.assert_array_equal(reward, [0.0, 1.0])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["captures"], ((0, 1),))
        self.assertEqual(info["captured_total"], 1)
        self.assertEqual(info["first_contact_step"], 1)
        with self.assertRaises(RuntimeError):
            env.step(stationary_policy(2))

    def test_horizon_truncates_without_capture(self) -> None:
        env = ParticleCollectorEnv(
            self._small_config(
                horizon=1,
                diffusion_sigma=0.0,
                signal_strength=0.0,
                collector_radius=0.0,
            )
        )
        env.reset(seed=3)
        _, _, terminated, truncated, _ = env.step(stationary_policy(2))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_zero_diffusion_zero_signal_stationary_limit(self) -> None:
        env = ParticleCollectorEnv(
            self._small_config(diffusion_sigma=0.0, signal_strength=0.0)
        )
        env.reset(seed=20)
        assert env.particle_positions is not None
        assert env.collector_positions is not None
        particles_before = env.particle_positions.copy()
        collectors_before = env.collector_positions.copy()
        env.step(stationary_policy(2))
        np.testing.assert_array_equal(env.particle_positions, particles_before)
        np.testing.assert_array_equal(env.collector_positions, collectors_before)

    def test_high_signal_changes_only_post_reset_particle_path(self) -> None:
        common = dict(
            particle_count=10,
            collector_count=2,
            horizon=2,
            nearest_particles_k=3,
            diffusion_sigma=0.0,
            field_family="uniform",
            field_kwargs={"orientation": 0.0},
        )
        null = ParticleCollectorEnv(ParticleEnvConfig(**common, signal_strength=0.0))
        signal = ParticleCollectorEnv(ParticleEnvConfig(**common, signal_strength=0.5))
        null.reset(seed=88)
        signal.reset(seed=88)
        np.testing.assert_array_equal(null.particle_positions, signal.particle_positions)
        np.testing.assert_array_equal(null.collector_positions, signal.collector_positions)
        action = stationary_policy(2)
        null.step(action)
        signal.step(action)
        self.assertFalse(np.array_equal(null.particle_positions, signal.particle_positions))
        np.testing.assert_array_equal(null.collector_positions, signal.collector_positions)

    def test_step_before_reset_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            ParticleCollectorEnv(self._small_config()).step(stationary_policy(2))

    def test_all_smoke_policies_step_the_default_environment(self) -> None:
        for policy_name in ("stationary", "random", "privileged"):
            env = ParticleCollectorEnv()
            env.reset(seed=101)
            if policy_name == "stationary":
                action = stationary_policy(env.config.collector_count)
            elif policy_name == "random":
                action = random_policy(
                    env.config.collector_count, np.random.default_rng(202)
                )
            else:
                assert env.collector_positions is not None
                action = privileged_field_policy(
                    env.collector_positions,
                    field_family=env.config.field_family,
                    signal_strength=env.config.signal_strength,
                    field_kwargs=env.config.field_kwargs,
                )
            observations, reward, terminated, truncated, info = env.step(action)
            self.assertEqual(len(observations), env.config.collector_count)
            self.assertEqual(reward.shape, (env.config.collector_count,))
            self.assertFalse(terminated and truncated)
            self.assertEqual(info["step"], 1)

    def test_newly_visible_velocity_is_invalid_until_next_visible_step(self) -> None:
        config = ParticleEnvConfig(
            particle_count=1,
            collector_count=1,
            horizon=3,
            dt=0.1,
            diffusion_sigma=0.0,
            sensing_radius=0.2,
            collector_radius=0.005,
            signal_strength=0.2,
            field_family="uniform",
            field_kwargs={"orientation": np.pi},
            nearest_particles_k=1,
        )
        env = ParticleCollectorEnv(config)
        env.reset(seed=12)
        assert env.particle_positions is not None
        assert env.collector_positions is not None
        env.particle_positions[0] = env.collector_positions[0] + [0.21, 0.0]
        env._last_visibility = env._visibility()

        first, _, _, _, _ = env.step(stationary_policy(1))
        self.assertTrue(first[0]["particle_mask"][0])
        self.assertFalse(first[0]["velocity_valid_mask"][0])
        np.testing.assert_array_equal(first[0]["particles"][0, 2:4], 0.0)

        second, _, _, _, _ = env.step(stationary_policy(1))
        self.assertTrue(second[0]["velocity_valid_mask"][0])
        np.testing.assert_allclose(second[0]["particles"][0, 2:4], [-0.2, 0.0])

    def test_zero_signal_full_rollout_identity_with_paired_actions(self) -> None:
        common = dict(
            particle_count=16,
            collector_count=4,
            horizon=5,
            nearest_particles_k=4,
            signal_strength=0.0,
        )
        null = ParticleCollectorEnv(ParticleEnvConfig(**common, field_family="null"))
        zero_signal = ParticleCollectorEnv(
            ParticleEnvConfig(**common, field_family="uniform")
        )
        null.reset(seed=303)
        zero_signal.reset(seed=303)
        actions = random_action_tensor(404, horizon=5, collector_count=4)
        for action in actions:
            null_result = null.step(action)
            signal_result = zero_signal.step(action)
            _assert_observations_equal(self, null_result[0], signal_result[0])
            np.testing.assert_array_equal(null_result[1], signal_result[1])
            self.assertEqual(null_result[2:], signal_result[2:])
            np.testing.assert_array_equal(
                null.particle_positions, zero_signal.particle_positions
            )
            np.testing.assert_array_equal(
                null.collector_positions, zero_signal.collector_positions
            )

    def test_high_signal_upstream_oracle_intercepts_before_stationary_microcase(self) -> None:
        config = ParticleEnvConfig(
            particle_count=1,
            collector_count=1,
            horizon=12,
            dt=0.1,
            diffusion_sigma=0.0,
            collector_max_speed=0.2,
            sensing_radius=1.0,
            collector_radius=0.015,
            signal_strength=0.2,
            field_family="uniform",
            field_kwargs={"orientation": 0.0},
            nearest_particles_k=1,
        )

        def run(use_oracle: bool) -> int:
            env = ParticleCollectorEnv(config)
            env.reset(seed=55)
            assert env.particle_positions is not None
            assert env.collector_positions is not None
            env.particle_positions[0] = [0.3, 0.5]
            env._last_visibility = env._visibility()
            while env.first_contact_step is None:
                if use_oracle:
                    action = privileged_field_policy(
                        env.collector_positions,
                        field_family=config.field_family,
                        signal_strength=config.signal_strength,
                        field_kwargs=config.field_kwargs,
                    )
                else:
                    action = stationary_policy(1)
                _, _, terminated, truncated, _ = env.step(action)
                self.assertFalse(truncated)
                if terminated:
                    break
            assert env.first_contact_step is not None
            return env.first_contact_step

        self.assertLess(run(use_oracle=True), run(use_oracle=False))


class TestSmokePolicies(unittest.TestCase):
    def test_stationary_and_random_actions_obey_action_contract(self) -> None:
        np.testing.assert_array_equal(stationary_policy(3), np.zeros((3, 2)))
        first = random_policy(1_000, np.random.default_rng(6))
        second = random_policy(1_000, np.random.default_rng(6))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(np.linalg.norm(first, axis=1) <= 1.0))

    def test_privileged_uniform_policy_reads_true_field(self) -> None:
        action = privileged_field_policy(
            [[0.1, 0.1], [0.8, 0.6]],
            field_family="uniform",
            signal_strength=2.0,
            field_kwargs={"orientation": np.pi / 2.0},
        )
        np.testing.assert_allclose(action, [[0.0, -1.0], [0.0, -1.0]], atol=1e-15)
        null_action = privileged_field_policy(
            [[0.1, 0.1]], field_family="null", signal_strength=0.0
        )
        np.testing.assert_array_equal(null_action, 0.0)

    def test_pregenerated_random_action_tensor_is_pair_identical(self) -> None:
        left = random_action_tensor(123, horizon=7, collector_count=4)
        right = random_action_tensor(123, horizon=7, collector_count=4)
        np.testing.assert_array_equal(left, right)
        self.assertEqual(left.shape, (7, 4, 2))
        self.assertTrue(np.all(np.linalg.norm(left, axis=2) <= 1.0))


if __name__ == "__main__":
    unittest.main()
