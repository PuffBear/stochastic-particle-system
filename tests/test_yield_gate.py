from __future__ import annotations

import unittest

import numpy as np

from analysis.run_yield_gate import (
    EVALUATION_STEPS,
    _stream_checksums,
    unique_capture_yield_through_step,
)
from particle_benchmark.environment import ParticleCollectorEnv, ParticleEnvConfig
from particle_benchmark.policies import stationary_policy


class TestFixedHorizonYieldEndpoint(unittest.TestCase):
    def test_unique_counting_deduplicates_particle_ids(self) -> None:
        events = [
            {"step": 1, "particle_id": 7},
            {"step": 3, "particle_id": 7},
            {"step": 3, "particle_id": 9},
        ]
        self.assertEqual(unique_capture_yield_through_step(events, cutoff_step=67), 2)

    def test_step_67_is_included_and_step_68_is_excluded(self) -> None:
        events = [
            {"step": 67, "particle_id": 1},
            {"step": 68, "particle_id": 2},
        ]
        self.assertEqual(unique_capture_yield_through_step(events, cutoff_step=67), 1)

    def test_environment_continues_after_first_contact(self) -> None:
        env = ParticleCollectorEnv(
            ParticleEnvConfig(
                particle_count=2,
                collector_count=1,
                horizon=3,
                dt=0.02,
                diffusion_sigma=0.0,
                collector_max_speed=0.0,
                collector_radius=0.012,
                sensing_radius=1.0,
                field_family="uniform",
                signal_strength=0.0,
                nearest_particles_k=2,
            )
        )
        env.reset(seed=81)
        assert env.particle_positions is not None
        assert env.collector_positions is not None
        env.particle_positions[0] = env.collector_positions[0]
        env.particle_positions[1] = [0.95, 0.95]
        env._last_visibility = env._visibility()
        _, _, terminated, truncated, info1 = env.step(stationary_policy(1))
        self.assertEqual(info1["first_contact_step"], 1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _, _, _, _, info2 = env.step(stationary_policy(1))
        self.assertEqual(info2["step"], 2)
        self.assertEqual(info2["captured_total"], 1)

    def test_matched_null_signal_stream_provenance(self) -> None:
        null_env = ParticleCollectorEnv(
            ParticleEnvConfig(horizon=EVALUATION_STEPS, signal_strength=0.0)
        )
        signal_env = ParticleCollectorEnv(
            ParticleEnvConfig(horizon=EVALUATION_STEPS, signal_strength=0.06)
        )
        null_env.reset(seed=2001)
        signal_env.reset(seed=2001)
        self.assertEqual(_stream_checksums(null_env), _stream_checksums(signal_env))
        np.testing.assert_array_equal(null_env._noise, signal_env._noise)


if __name__ == "__main__":
    unittest.main()
