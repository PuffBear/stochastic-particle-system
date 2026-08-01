from __future__ import annotations

import unittest

import numpy as np

from particle_benchmark.metrics.inference import (
    grid_censored_boundary,
    persistent_grid_censored_boundary,
    simultaneous_paired_lower_bounds,
)


class TestSimultaneousInference(unittest.TestCase):
    def test_reproducible_and_row_blocked(self) -> None:
        effects = np.array(
            [[0.1, 0.2], [0.0, 0.3], [0.2, 0.4], [0.1, 0.5]], dtype=float
        )
        left = simultaneous_paired_lower_bounds(
            effects, bootstrap_draws=1000, bootstrap_seed=9
        )
        right = simultaneous_paired_lower_bounds(
            effects, bootstrap_draws=1000, bootstrap_seed=9
        )
        np.testing.assert_array_equal(left.mean, right.mean)
        np.testing.assert_array_equal(left.lower, right.lower)
        self.assertGreaterEqual(left.critical_value, 0.0)

    def test_row_permutation_is_bitwise_invariant(self) -> None:
        effects = np.array(
            [[0.1, 0.2], [-0.1, 0.3], [0.4, -0.2], [0.0, 0.1]], dtype=float
        )
        left = simultaneous_paired_lower_bounds(effects, bootstrap_draws=1000)
        right = simultaneous_paired_lower_bounds(effects[[2, 0, 3, 1]], bootstrap_draws=1000)
        np.testing.assert_array_equal(left.lower, right.lower)

    def test_constant_positive_effect_has_exact_positive_bound(self) -> None:
        effects = np.full((8, 3), 0.25)
        result = simultaneous_paired_lower_bounds(effects, bootstrap_draws=500)
        np.testing.assert_allclose(result.lower, 0.25)

    def test_boundary_reports_interval_and_censoring(self) -> None:
        self.assertEqual(
            grid_censored_boundary([0.1, 0.2, 0.4], [-0.1, 0.01, 0.02]),
            {"status": "interval", "lower_rho": 0.1, "upper_rho": 0.2},
        )
        self.assertEqual(
            grid_censored_boundary([0.1, 0.2], [-0.1, 0.0])["status"],
            "right_censored",
        )
        self.assertEqual(
            grid_censored_boundary([0.1, 0.2], [0.01, 0.02])["status"],
            "left_censored",
        )

    def test_zero_fixture_never_crosses(self) -> None:
        result = simultaneous_paired_lower_bounds(np.zeros((12, 5)), bootstrap_draws=500)
        np.testing.assert_array_equal(result.lower, 0.0)
        self.assertEqual(
            grid_censored_boundary([0.1, 0.25, 0.5, 1.0, 2.0], result.lower)["status"],
            "right_censored",
        )

    def test_first_and_persistent_crossings_differ(self) -> None:
        rho = [0.1, 0.25, 0.5, 1.0, 2.0]
        lower = [-0.1, 0.01, -0.02, 0.03, 0.04]
        self.assertEqual(grid_censored_boundary(rho, lower)["upper_rho"], 0.25)
        self.assertEqual(
            persistent_grid_censored_boundary(rho, lower)["upper_rho"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
