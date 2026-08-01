from __future__ import annotations

import unittest
import numpy as np

from particle_benchmark.refinement import aggregate_brownian_increments, coupled_standard_normals


class TestCoupledRefinement(unittest.TestCase):
    def test_exact_increment_aggregation(self) -> None:
        fine = np.arange(24, dtype=float).reshape(6, 2, 2)
        coarse = aggregate_brownian_increments(fine, 3)
        np.testing.assert_array_equal(coarse[0], fine[:3].sum(axis=0))
        np.testing.assert_array_equal(coarse[1], fine[3:].sum(axis=0))

    def test_generated_paths_have_exact_same_endpoint(self) -> None:
        fine, coarse = coupled_standard_normals(7, finest_steps=12, shape_per_step=(3, 2), factor=3)
        fine_endpoint = (fine / np.sqrt(12.0)).sum(axis=0)
        coarse_endpoint = (coarse / np.sqrt(4.0)).sum(axis=0)
        np.testing.assert_allclose(fine_endpoint, coarse_endpoint, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
