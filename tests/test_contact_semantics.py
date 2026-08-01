from __future__ import annotations

import unittest

import numpy as np

from particle_benchmark.dynamics.capture import (
    CaptureState,
    resolve_swept_fixed_captures,
)
from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.policies import stationary_policy


class TestSweptFixedContact(unittest.TestCase):
    def _resolve(
        self,
        particle_start: list[list[float]],
        particle_end: list[list[float]],
        collector_start: list[list[float]],
        collector_end: list[list[float]],
        **kwargs: object,
    ):
        state = CaptureState.empty(len(particle_start), len(collector_start))
        result = resolve_swept_fixed_captures(
            particle_start,
            particle_end,
            collector_start,
            collector_end,
            state,
            collector_radius=0.1,
            particle_radius=0.0,
            event_seed=17,
            step_index=3,
            **kwargs,
        )
        return state, result

    def test_crossing_is_detected_when_endpoints_do_not_overlap(self) -> None:
        state, result = self._resolve(
            [[0.0, 0.5]], [[1.0, 0.5]], [[0.5, 0.6]], [[0.5, 0.6]]
        )
        self.assertEqual(len(result.events), 1)
        self.assertAlmostEqual(result.events[0].time_fraction, 0.5)
        self.assertEqual(result.events[0].collector_id, 0)
        np.testing.assert_array_equal(state.owner, [0])

    def test_near_miss_is_not_captured(self) -> None:
        state, result = self._resolve(
            [[0.0, 0.5]], [[1.0, 0.5]], [[0.5, 0.6001]], [[0.5, 0.6001]]
        )
        self.assertEqual(result.events, ())
        np.testing.assert_array_equal(state.owner, [-1])

    def test_endpoint_contact_regression(self) -> None:
        _, result = self._resolve(
            [[0.3, 0.5]], [[0.4, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]
        )
        self.assertEqual(len(result.events), 1)
        self.assertAlmostEqual(result.events[0].time_fraction, 1.0)

    def test_earliest_contact_wins_before_collector_id(self) -> None:
        _, result = self._resolve(
            [[0.0, 0.5]],
            [[1.0, 0.5]],
            [[0.3, 0.5], [0.7, 0.5]],
            [[0.3, 0.5], [0.7, 0.5]],
        )
        self.assertEqual(result.events[0].collector_id, 0)
        self.assertAlmostEqual(result.events[0].time_fraction, 0.2)

    def test_exact_tie_is_event_keyed_and_collector_order_invariant(self) -> None:
        particle_start = [[0.0, 0.5]]
        particle_end = [[1.0, 0.5]]
        collectors = [[0.5, 0.4], [0.5, 0.6]]
        _, canonical = self._resolve(
            particle_start, particle_end, collectors, collectors
        )
        _, permuted = self._resolve(
            particle_start,
            particle_end,
            collectors[::-1],
            collectors[::-1],
            collector_ids=[1, 0],
        )
        self.assertEqual(
            canonical.events[0].collector_id, permuted.events[0].collector_id
        )
        self.assertEqual(canonical.tie_decisions, permuted.tie_decisions)
        self.assertEqual(len(canonical.tie_decisions), 1)
        self.assertEqual(
            canonical.tie_decisions[0].candidate_collector_ids, (0, 1)
        )

    def test_particle_iteration_order_does_not_shift_tie_decisions(self) -> None:
        particles_start = [[0.0, 0.5], [1.0, 0.5]]
        particles_end = [[1.0, 0.5], [0.0, 0.5]]
        collectors = [[0.5, 0.4], [0.5, 0.6]]
        _, canonical = self._resolve(
            particles_start,
            particles_end,
            collectors,
            collectors,
            particle_ids=[10, 20],
        )
        _, permuted = self._resolve(
            particles_start[::-1],
            particles_end[::-1],
            collectors,
            collectors,
            particle_ids=[20, 10],
        )
        canonical_choices = {
            event.particle_id: event.collector_id for event in canonical.events
        }
        permuted_choices = {
            event.particle_id: event.collector_id for event in permuted.events
        }
        self.assertEqual(canonical_choices, permuted_choices)

    def test_reflected_path_contact_uses_unfolded_proposal(self) -> None:
        state, result = self._resolve(
            [[0.9, 0.5]],
            [[0.7, 0.5]],
            [[0.7, 0.5]],
            [[0.7, 0.5]],
            particle_reflected=[True],
            particle_unfolded_end_positions=[[1.3, 0.5]],
            collector_unfolded_end_positions=[[0.7, 0.5]],
            arena_size=(1.0, 1.0),
        )
        self.assertEqual(len(result.events), 1)
        self.assertAlmostEqual(result.events[0].time_fraction, 0.75)
        self.assertEqual(result.guarded_reflection_pairs, 0)
        np.testing.assert_array_equal(state.owner, [0])

    def test_arbitrary_multiwall_overshoot_uses_earliest_segment(self) -> None:
        _, result = self._resolve(
            [[0.2, 0.5]],
            [[0.8, 0.5]],
            [[0.5, 0.5]],
            [[0.5, 0.5]],
            particle_reflected=[True],
            particle_unfolded_end_positions=[[3.2, 0.5]],
            collector_unfolded_end_positions=[[0.5, 0.5]],
            arena_size=(1.0, 1.0),
        )
        self.assertAlmostEqual(result.events[0].time_fraction, 1.0 / 15.0)

    def test_both_particle_and_collector_can_move(self) -> None:
        _, result = self._resolve(
            [[0.1, 0.5]],
            [[0.9, 0.5]],
            [[0.9, 0.5]],
            [[0.1, 0.5]],
            particle_unfolded_end_positions=[[0.9, 0.5]],
            collector_unfolded_end_positions=[[0.1, 0.5]],
            arena_size=(1.0, 1.0),
        )
        self.assertAlmostEqual(result.events[0].time_fraction, 0.4375)

    def test_reflecting_collector_path_is_segmented_too(self) -> None:
        _, result = self._resolve(
            [[0.7, 0.5]],
            [[0.7, 0.5]],
            [[0.9, 0.5]],
            [[0.7, 0.5]],
            collector_reflected=[True],
            particle_unfolded_end_positions=[[0.7, 0.5]],
            collector_unfolded_end_positions=[[1.3, 0.5]],
            arena_size=(1.0, 1.0),
        )
        self.assertAlmostEqual(result.events[0].time_fraction, 0.75)

    def test_declared_reflection_without_unfolded_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unfolded endpoints"):
            self._resolve(
                [[0.9, 0.5]],
                [[0.7, 0.5]],
                [[0.7, 0.5]],
                [[0.7, 0.5]],
                particle_reflected=[True],
            )

    def test_simultaneous_corner_reflection_is_one_breakpoint(self) -> None:
        _, result = self._resolve(
            [[0.8, 0.8]],
            [[0.6, 0.6]],
            [[0.7, 0.7]],
            [[0.7, 0.7]],
            particle_reflected=[True],
            particle_unfolded_end_positions=[[1.4, 1.4]],
            collector_unfolded_end_positions=[[0.7, 0.7]],
            arena_size=(1.0, 1.0),
        )
        self.assertGreater(result.events[0].time_fraction, 1.0 / 3.0)
        expected = (1.2 - (0.7 + 0.1 / np.sqrt(2.0))) / 0.6
        self.assertAlmostEqual(result.events[0].time_fraction, expected)

    def test_linear_chord_contact_time_is_dt_split_consistent(self) -> None:
        _, full = self._resolve(
            [[0.0, 0.5]], [[1.0, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]
        )
        _, half = self._resolve(
            [[0.0, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]
        )
        full_physical_time = full.events[0].time_fraction
        half_physical_time = 0.5 * half.events[0].time_fraction
        self.assertAlmostEqual(full_physical_time, half_physical_time)

    def test_reflected_contact_time_is_dt_split_consistent(self) -> None:
        _, full = self._resolve(
            [[0.8, 0.8]],
            [[0.6, 0.6]],
            [[0.7, 0.7]],
            [[0.7, 0.7]],
            particle_unfolded_end_positions=[[1.4, 1.4]],
            collector_unfolded_end_positions=[[0.7, 0.7]],
            arena_size=(1.0, 1.0),
        )
        _, first_half = self._resolve(
            [[0.8, 0.8]],
            [[0.9, 0.9]],
            [[0.7, 0.7]],
            [[0.7, 0.7]],
            particle_unfolded_end_positions=[[1.1, 1.1]],
            collector_unfolded_end_positions=[[0.7, 0.7]],
            arena_size=(1.0, 1.0),
        )
        self.assertEqual(first_half.events, ())
        _, second_half = self._resolve(
            [[0.9, 0.9]],
            [[0.6, 0.6]],
            [[0.7, 0.7]],
            [[0.7, 0.7]],
            particle_unfolded_end_positions=[[0.6, 0.6]],
            collector_unfolded_end_positions=[[0.7, 0.7]],
            arena_size=(1.0, 1.0),
        )
        split_physical_time = 0.5 + 0.5 * second_half.events[0].time_fraction
        self.assertAlmostEqual(full.events[0].time_fraction, split_physical_time)


class TestEnvironmentContactIntegration(unittest.TestCase):
    def test_environment_captures_within_step_crossing(self) -> None:
        env = ParticleCollectorEnv(
            ParticleEnvConfig(
                particle_count=1,
                collector_count=1,
                horizon=2,
                dt=0.5,
                diffusion_sigma=0.0,
                collector_max_speed=0.0,
                collector_radius=0.01,
                sensing_radius=1.0,
                field_family="uniform",
                signal_strength=0.8,
                field_kwargs={"orientation": 0.0},
                nearest_particles_k=1,
            )
        )
        reset_info = env.reset(seed=71)[1]
        assert env.particle_positions is not None
        env.particle_positions[0] = [0.3, 0.5]
        env._last_visibility = env._visibility()
        _, reward, terminated, _, info = env.step(stationary_policy(1))
        self.assertTrue(terminated)
        np.testing.assert_array_equal(reward, [1.0])
        self.assertAlmostEqual(info["contact_time_fractions"][0], 0.475)
        self.assertEqual(info["first_contact_step"], 1)
        self.assertEqual(reset_info["scenario_seed"], 71)
        self.assertEqual(info["tie_scheme"], "event_keyed_seed_step_particle_v1")

    def test_environment_uses_exact_multiwall_reflected_path(self) -> None:
        env = ParticleCollectorEnv(
            ParticleEnvConfig(
                particle_count=1,
                collector_count=1,
                horizon=2,
                dt=1.0,
                diffusion_sigma=0.0,
                collector_max_speed=0.0,
                collector_radius=0.01,
                sensing_radius=1.0,
                field_family="uniform",
                signal_strength=1.2,
                field_kwargs={"orientation": 0.0},
                nearest_particles_k=1,
            )
        )
        env.reset(seed=72)
        assert env.particle_positions is not None
        env.particle_positions[0] = [0.9, 0.5]
        env._last_visibility = env._visibility()
        _, _, terminated, _, info = env.step(stationary_policy(1))
        self.assertTrue(terminated)
        self.assertEqual(info["guarded_reflection_pairs"], 0)
        self.assertEqual(
            info["contact_model"], "fixed_piecewise_specular_reflection_exact_v1"
        )
        self.assertIn("dt audit", info["continuous_contact_limitation"])


if __name__ == "__main__":
    unittest.main()
