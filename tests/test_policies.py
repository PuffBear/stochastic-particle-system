from __future__ import annotations

import unittest

import numpy as np

from particle_benchmark.policies import (
    coverage_policy,
    density_greedy_policy,
    local_flow_v1_policy,
)


def _observation(
    *, relative: list[list[float]], velocities: list[list[float]], valid: list[bool]
) -> dict[str, np.ndarray]:
    slots = np.zeros((len(relative), 5), dtype=np.float64)
    slots[:, :2] = relative
    slots[:, 2:4] = velocities
    return {
        "self_position": np.array([0.5, 0.5]),
        "particles": slots,
        "particle_mask": np.ones(len(relative), dtype=np.bool_),
        "velocity_valid_mask": np.asarray(valid, dtype=np.bool_),
        "teammate_relative_positions": np.zeros((0, 2)),
    }


class TestFrozenPolicies(unittest.TestCase):
    def test_local_flow_moves_against_valid_mean_only(self) -> None:
        observation = _observation(
            relative=[[1.0, 0.0], [0.0, 1.0]],
            velocities=[[2.0, 0.0], [100.0, 100.0]],
            valid=[True, False],
        )
        np.testing.assert_allclose(local_flow_v1_policy((observation,)), [[-1.0, 0.0]])

    def test_local_flow_stops_without_causal_velocity(self) -> None:
        observation = _observation(
            relative=[[1.0, 0.0]], velocities=[[2.0, 0.0]], valid=[False]
        )
        np.testing.assert_array_equal(local_flow_v1_policy((observation,)), 0.0)

    def test_density_greedy_does_not_use_velocity(self) -> None:
        left = _observation(
            relative=[[1.0, 0.0]], velocities=[[9.0, -7.0]], valid=[True]
        )
        right = _observation(
            relative=[[1.0, 0.0]], velocities=[[-3.0, 8.0]], valid=[False]
        )
        np.testing.assert_array_equal(
            density_greedy_policy((left,)), density_greedy_policy((right,))
        )

    def test_coverage_is_particle_independent_and_bounded(self) -> None:
        left = _observation(
            relative=[[1.0, 0.0]], velocities=[[1.0, 0.0]], valid=[True]
        )
        right = _observation(
            relative=[[-1.0, 0.0]], velocities=[[-1.0, 0.0]], valid=[False]
        )
        action_left = coverage_policy((left,), step=0)
        action_right = coverage_policy((right,), step=0)
        np.testing.assert_array_equal(action_left, action_right)
        self.assertLessEqual(float(np.linalg.norm(action_left[0])), 1.0)


if __name__ == "__main__":
    unittest.main()
