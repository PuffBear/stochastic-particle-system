from __future__ import annotations

import unittest

import numpy as np

from particle_benchmark.dynamics.fields import (
    EpisodeFrozenGaussianField,
    field_velocity,
)


class TestEpisodeFrozenGaussianField(unittest.TestCase):
    def test_realization_is_repeatable_read_only_and_periodic(self) -> None:
        field = EpisodeFrozenGaussianField.sample(
            seed=17, correlation_length=0.2, max_frequency=4
        )
        positions = np.array([[0.12, 0.31], [0.71, 0.83]])
        first = field.velocity(positions)
        np.testing.assert_array_equal(first, field.velocity(positions))
        np.testing.assert_allclose(first, field.velocity(positions + 1.0), atol=1e-12)
        self.assertFalse(field.cosine_coefficients.flags.writeable)
        self.assertFalse(field.sine_coefficients.flags.writeable)

    def test_declared_marginal_variance_is_invariant_to_correlation_length(self) -> None:
        for ell_c in (0.05, 0.15, 0.4, 1.0):
            field = EpisodeFrozenGaussianField.sample(
                seed=3,
                correlation_length=ell_c,
                component_variance=0.7,
                max_frequency=6,
            )
            self.assertAlmostEqual(
                float(field.component_covariance([0.0, 0.0])), 0.7, places=12
            )
            self.assertAlmostEqual(
                field.normalized_constant_weight
                + float(np.sum(field.normalized_mode_weights)),
                1.0,
                places=12,
            )

    def test_longer_declared_length_increases_nearby_correlation(self) -> None:
        short = EpisodeFrozenGaussianField.sample(
            seed=3, correlation_length=0.08, max_frequency=6
        )
        long = EpisodeFrozenGaussianField.sample(
            seed=3, correlation_length=0.35, max_frequency=6
        )
        displacement = np.array([0.15, 0.0])
        short_correlation = float(short.component_covariance(displacement))
        long_correlation = float(long.component_covariance(displacement))
        self.assertGreater(long_correlation, short_correlation)

    def test_empirical_ensemble_recovers_declared_covariance(self) -> None:
        point_a = np.array([0.17, 0.23])
        point_b = np.array([0.31, 0.27])
        prototype = EpisodeFrozenGaussianField.sample(
            seed=0,
            correlation_length=0.18,
            component_variance=0.6,
            max_frequency=4,
        )
        target = float(prototype.component_covariance(point_a - point_b))
        # This sanity check uses 48 realizations, leaving room for every other
        # field microcase under SPS-WO-09's total cap of 64.  The exact
        # covariance identity is checked separately above, so this
        # intentionally uses a wider Monte Carlo tolerance rather than
        # silently increasing the work-order cap.
        samples = np.empty((48, 2, 2), dtype=np.float64)
        for seed in range(samples.shape[0]):
            realization = EpisodeFrozenGaussianField.sample(
                seed=seed,
                correlation_length=0.18,
                component_variance=0.6,
                max_frequency=4,
            )
            samples[seed] = realization.velocity([point_a, point_b])
        empirical_x = float(np.mean(samples[:, 0, 0] * samples[:, 1, 0]))
        empirical_y = float(np.mean(samples[:, 0, 1] * samples[:, 1, 1]))
        empirical_pooled = float(
            np.mean(samples[:, 0, :] * samples[:, 1, :])
        )
        cross_component = float(np.mean(samples[:, 0, 0] * samples[:, 1, 1]))
        self.assertLess(abs(empirical_pooled - target), 0.08)
        self.assertLess(abs(empirical_x - target), 0.25)
        self.assertLess(abs(empirical_y - target), 0.25)
        self.assertLess(abs(cross_component), 0.25)

    def test_realization_checksum_is_canonical_and_parameter_sensitive(self) -> None:
        common = dict(correlation_length=0.2, max_frequency=4)
        first = EpisodeFrozenGaussianField.sample(seed=41, **common)
        repeated = EpisodeFrozenGaussianField.sample(seed=41, **common)
        changed_seed = EpisodeFrozenGaussianField.sample(seed=42, **common)
        changed_length = EpisodeFrozenGaussianField.sample(
            seed=41, correlation_length=0.21, max_frequency=4
        )
        self.assertRegex(first.realization_sha256(), r"^[0-9a-f]{64}$")
        self.assertEqual(first.realization_sha256(), repeated.realization_sha256())
        self.assertNotEqual(
            first.realization_sha256(), changed_seed.realization_sha256()
        )
        self.assertNotEqual(
            first.realization_sha256(), changed_length.realization_sha256()
        )

    def test_field_velocity_requires_and_uses_frozen_realization(self) -> None:
        positions = np.array([[0.1, 0.2], [0.4, 0.8]])
        with self.assertRaisesRegex(ValueError, "episode-frozen"):
            field_velocity(positions, "periodic_gaussian", 0.5)
        field = EpisodeFrozenGaussianField.sample(
            seed=9, correlation_length=0.25, max_frequency=3
        )
        np.testing.assert_allclose(
            field_velocity(
                positions,
                "periodic_gaussian",
                0.5,
                frozen_field=field,
            ),
            0.5 * field.velocity(positions),
        )

    def test_invalid_contract_parameters_are_rejected(self) -> None:
        for invalid in (0.0, -0.1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                EpisodeFrozenGaussianField.sample(
                    seed=1, correlation_length=invalid
                )


if __name__ == "__main__":
    unittest.main()
