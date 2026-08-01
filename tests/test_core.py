from __future__ import annotations

import unittest

import numpy as np

from particle_benchmark.dynamics.boundaries import reflect
from particle_benchmark.dynamics.capture import CaptureState, resolve_captures
from particle_benchmark.dynamics.fields import field_velocity
from particle_benchmark.dynamics.particles import simulate_matched_pair
from particle_benchmark.metrics.episode import (
    first_interception_step,
    matched_first_interception_gain,
    per_step_signal_to_noise,
)
from particle_benchmark.seeding import brownian_tensor, make_streams


class TestSeeding(unittest.TestCase):
    def test_same_seed_is_bitwise_identical(self) -> None:
        left = brownian_tensor(7, horizon=8, particle_count=16)
        right = brownian_tensor(7, horizon=8, particle_count=16)
        self.assertTrue(np.array_equal(left, right))

    def test_different_seed_changes_noise(self) -> None:
        left = brownian_tensor(7, horizon=8, particle_count=16)
        right = brownian_tensor(8, horizon=8, particle_count=16)
        self.assertFalse(np.array_equal(left, right))

    def test_field_toggle_cannot_change_noise_stream(self) -> None:
        noise_before = brownian_tensor(19, horizon=3, particle_count=4)
        streams = make_streams(19)
        _ = streams.field.uniform()
        noise_after = brownian_tensor(19, horizon=3, particle_count=4)
        self.assertTrue(np.array_equal(noise_before, noise_after))


class TestBoundaries(unittest.TestCase):
    def test_reflection_preserves_overshoot(self) -> None:
        actual = reflect(np.array([-0.2, 1.2, 2.2, -2.2]), 1.0)
        np.testing.assert_allclose(actual, [0.2, 0.8, 0.2, 0.2])

    def test_rectangular_broadcast(self) -> None:
        actual = reflect([[1.2, -0.25]], [1.0, 2.0])
        np.testing.assert_allclose(actual, [[0.8, 0.25]])


class TestFields(unittest.TestCase):
    def test_null_is_zero(self) -> None:
        pos = np.array([[0.1, 0.2], [0.8, 0.9]])
        np.testing.assert_array_equal(field_velocity(pos, "null", 9.0), 0.0)

    def test_uniform_magnitude_is_constant(self) -> None:
        pos = np.zeros((5, 2))
        vel = field_velocity(pos, "uniform", 0.3, orientation=0.7)
        np.testing.assert_allclose(np.linalg.norm(vel, axis=1), 0.3)

    def test_vortex_is_tangential_and_finite_at_centre(self) -> None:
        pos = np.array([[0.75, 0.5], [0.5, 0.75], [0.5, 0.5]])
        vel = field_velocity(pos, "vortex", 0.4, centre=(0.5, 0.5))
        radial = pos[:2] - np.array([0.5, 0.5])
        dot = np.sum(radial * vel[:2], axis=1)
        np.testing.assert_allclose(dot, 0.0, atol=1e-12)
        self.assertTrue(np.all(np.isfinite(vel)))
        np.testing.assert_array_equal(vel[2], 0.0)


class TestMatchedCounterfactuals(unittest.TestCase):
    def test_zero_signal_paths_are_identical(self) -> None:
        initial = np.array([[0.2, 0.3], [0.8, 0.7]])
        noise = brownian_tensor(41, horizon=5, particle_count=2)
        pair = simulate_matched_pair(
            initial,
            noise,
            dt=0.02,
            diffusion_sigma=0.06,
            arena_size=(1.0, 1.0),
            field_family="uniform",
            signal_strength=0.0,
            field_kwargs={"orientation": 1.2},
        )
        np.testing.assert_array_equal(pair.null, pair.signal)

    def test_signal_toggle_preserves_initial_state_and_noise(self) -> None:
        initial = np.array([[0.2, 0.3], [0.8, 0.7]])
        noise = brownian_tensor(42, horizon=5, particle_count=2)
        pair = simulate_matched_pair(
            initial,
            noise,
            dt=0.02,
            diffusion_sigma=0.06,
            arena_size=(1.0, 1.0),
            field_family="uniform",
            signal_strength=0.2,
            field_kwargs={"orientation": 0.0},
        )
        np.testing.assert_array_equal(pair.null[0], pair.signal[0])
        np.testing.assert_array_equal(pair.noise, noise)
        self.assertFalse(np.array_equal(pair.null[1:], pair.signal[1:]))

    def test_stationary_particles_remain_stationary(self) -> None:
        initial = np.array([[0.2, 0.3], [0.8, 0.7]])
        noise = np.zeros((4, 2, 2))
        pair = simulate_matched_pair(
            initial,
            noise,
            dt=0.02,
            diffusion_sigma=0.0,
            arena_size=(1.0, 1.0),
            field_family="null",
            signal_strength=0.0,
        )
        expected = np.broadcast_to(initial, pair.null.shape)
        np.testing.assert_array_equal(pair.null, expected)
        np.testing.assert_array_equal(pair.signal, expected)


class TestCapture(unittest.TestCase):
    def test_fixed_capture_is_permanent_and_single_owner(self) -> None:
        particles = np.array([[0.11, 0.1], [0.9, 0.9]])
        collectors = np.array([[0.1, 0.1], [0.9, 0.8]])
        state = CaptureState.empty(2, 2)
        rng = np.random.default_rng(1)
        first = resolve_captures(
            particles,
            collectors,
            state,
            geometry="fixed",
            collector_radius=0.02,
            particle_radius=0.0,
            tie_rng=rng,
        )
        second = resolve_captures(
            particles,
            collectors,
            state,
            geometry="fixed",
            collector_radius=0.02,
            particle_radius=0.0,
            tie_rng=rng,
        )
        self.assertEqual(first, [(0, 0)])
        self.assertEqual(second, [])
        np.testing.assert_array_equal(state.owner, [0, -1])

    def test_growing_node_captures_on_next_step(self) -> None:
        particles = np.array([[0.12, 0.1], [0.14, 0.1]])
        collectors = np.array([[0.1, 0.1]])
        state = CaptureState.empty(2, 1)
        rng = np.random.default_rng(2)
        first = resolve_captures(
            particles,
            collectors,
            state,
            geometry="growing",
            collector_radius=0.025,
            particle_radius=0.0,
            tie_rng=rng,
        )
        second = resolve_captures(
            particles,
            collectors,
            state,
            geometry="growing",
            collector_radius=0.025,
            particle_radius=0.0,
            tie_rng=rng,
        )
        self.assertEqual(first, [(0, 0)])
        self.assertEqual(second, [(1, 0)])
        self.assertEqual(len(state.offsets[0]), 2)


class TestMetrics(unittest.TestCase):
    def test_first_interception(self) -> None:
        self.assertEqual(first_interception_step([0, 0, 2, 0]), 3)
        self.assertIsNone(first_interception_step([0, 0]))

    def test_matched_gain_positive_when_signal_is_earlier(self) -> None:
        gain = matched_first_interception_gain([2, None], [5, None], horizon=10)
        np.testing.assert_allclose(gain, [3 / 10, 0.0])

    def test_metric_microcases(self) -> None:
        gain = matched_first_interception_gain(
            [1, 10, None],
            [None, 10, None],
            horizon=10,
        )
        np.testing.assert_allclose(gain, [1.0, 0.0, 0.0])

    def test_dimensionless_signal_to_noise(self) -> None:
        actual = per_step_signal_to_noise(0.2, dt=0.02, diffusion_sigma=0.06)
        self.assertAlmostEqual(actual, 0.2 * np.sqrt(0.02) / 0.06)


if __name__ == "__main__":
    unittest.main()
