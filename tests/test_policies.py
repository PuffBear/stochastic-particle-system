from __future__ import annotations

import unittest

import numpy as np

from particle_benchmark.policies import (
    bounded_team_velocity_summary,
    capacity_matched_velocity_controller,
    coverage_policy,
    density_greedy_policy,
    full_state_interception_oracle,
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
    def test_full_state_oracle_intercepts_and_rejects_impossible_target(self) -> None:
        intercept = full_state_interception_oracle(
            [[0.0, 0.0]], [[0.3, 0.0]], [[0.02, 0.0]], [True],
            collector_max_speed=0.12, receding_horizon=5.0,
        )
        np.testing.assert_allclose(intercept, [[1.0, 0.0]])
        impossible = full_state_interception_oracle(
            [[0.0, 0.0]], [[0.3, 0.0]], [[0.2, 0.0]], [True],
            collector_max_speed=0.12, receding_horizon=5.0,
        )
        np.testing.assert_array_equal(impossible, 0.0)

    def test_team_summary_is_permutation_invariant(self) -> None:
        a = _observation(relative=[[1, 0]], velocities=[[0.2, 0.4]], valid=[True])
        b = _observation(relative=[[0, 1]], velocities=[[-0.1, 0.2]], valid=[True])
        np.testing.assert_allclose(
            bounded_team_velocity_summary((a, b)),
            bounded_team_velocity_summary((b, a)),
        )
        self.assertTrue(np.all(np.abs(bounded_team_velocity_summary((a, b))) <= 1.0))

    def test_summary_has_no_masked_particle_leakage(self) -> None:
        visible = _observation(relative=[[1, 0], [0, 0]], velocities=[[0.2, 0.4], [0, 0]], valid=[True, False])
        altered = _observation(relative=[[1, 0], [999, -999]], velocities=[[0.2, 0.4], [999, 999]], valid=[True, True])
        visible["particle_mask"][1] = False
        altered["particle_mask"][1] = False
        np.testing.assert_allclose(
            bounded_team_velocity_summary((visible,)),
            bounded_team_velocity_summary((altered,)),
        )

    def test_controllers_have_identical_shape_and_agent_equivariance(self) -> None:
        a = _observation(relative=[[1, 0]], velocities=[[0.2, 0.4]], valid=[True])
        b = _observation(relative=[[0, 1]], velocities=[[-0.1, 0.2]], valid=[True])
        for shared in (False, True):
            original = capacity_matched_velocity_controller((a, b), shared=shared)
            permuted = capacity_matched_velocity_controller((b, a), shared=shared)
            self.assertEqual(original.shape, (2, 2))
            np.testing.assert_allclose(original, permuted[::-1])

    def test_constant_velocity_microcase_has_identical_actions(self) -> None:
        observations = tuple(
            _observation(
                relative=[[1.0, 0.0]],
                velocities=[[0.6, -0.2]],
                valid=[True],
            )
            for _ in range(4)
        )
        independent = capacity_matched_velocity_controller(
            observations, shared=False
        )
        global_mean = capacity_matched_velocity_controller(
            observations, shared=True
        )
        np.testing.assert_array_equal(independent, global_mean)

    def test_opposing_regions_preserve_local_actions_but_global_mean_stalls(self) -> None:
        observations = tuple(
            _observation(
                relative=[[1.0, 0.0]], velocities=[[velocity, 0.0]], valid=[True]
            )
            for velocity in (1.0, 1.0, -1.0, -1.0)
        )
        independent = capacity_matched_velocity_controller(
            observations, shared=False
        )
        global_mean = capacity_matched_velocity_controller(
            observations, shared=True
        )
        np.testing.assert_array_equal(independent[:, 0], [-1.0, -1.0, 1.0, 1.0])
        np.testing.assert_array_equal(global_mean, 0.0)

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
    full_state_interception_oracle,
