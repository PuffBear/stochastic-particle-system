"""Tests for the theoretical analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from particle_benchmark.theory.matching import (
    matched_variance_reduction,
    power_from_matching,
)
from particle_benchmark.theory.estimation import (
    apparent_velocity_noise_std,
    fisher_information_field_direction,
    team_sensing_summary,
)
from particle_benchmark.theory.dilution import (
    overlap_probability,
    expected_unique_coverage,
    snr_per_seed,
    dilution_snr_ratio,
)


# ---------------------------------------------------------------------------
# matching.py
# ---------------------------------------------------------------------------

class TestMatchedVarianceReduction:

    def test_perfectly_correlated_yields_zero_matched_variance(self):
        # If Y_signal = Y_null + constant, matched variance = 0.
        yn = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ys = yn + 1.0
        result = matched_variance_reduction(ys, yn)
        assert result["matched_variance"] == pytest.approx(0.0, abs=1e-12)
        assert result["variance_reduction_factor"] == pytest.approx(1.0, abs=1e-6)
        assert result["within_seed_correlation"] == pytest.approx(1.0, abs=1e-10)

    def test_independent_yields_no_reduction(self):
        # If Y_signal and Y_null are uncorrelated, matched ≈ unmatched.
        rng = np.random.default_rng(42)
        ys = rng.standard_normal(1000)
        yn = rng.standard_normal(1000)  # independent
        result = matched_variance_reduction(ys, yn)
        # With independent draws, reduction should be near zero
        assert abs(result["variance_reduction_factor"]) < 0.15

    def test_reduction_factor_equals_correlation_when_homoskedastic(self):
        # When Var(Y_s) == Var(Y_n), reduction = ρ_YY.
        # Construct: ys = base + e1, yn = base + e2, same noise level → equal variances.
        rng = np.random.default_rng(7)
        base = rng.standard_normal(5000)
        ys = base + rng.standard_normal(5000) * 0.5
        yn = base + rng.standard_normal(5000) * 0.5
        result = matched_variance_reduction(ys, yn)
        corr = result["within_seed_correlation"]
        reduction = result["variance_reduction_factor"]
        # In the equal-variance case, reduction = 2ρσ²/(2σ²) = ρ
        assert reduction == pytest.approx(corr, abs=0.03)

    def test_matched_variance_formula(self):
        ys = np.array([2.0, 4.0, 6.0])
        yn = np.array([1.0, 3.0, 5.0])
        result = matched_variance_reduction(ys, yn)
        # D = [1, 1, 1], Var(D) = 0
        assert result["matched_variance"] == pytest.approx(0.0, abs=1e-12)
        assert result["mean_difference"] == pytest.approx(1.0, abs=1e-12)

    def test_effective_n_equivalent_is_inverse_of_matched_variance_fraction(self):
        rng = np.random.default_rng(99)
        ys = rng.standard_normal(50) + 1.0
        yn = 0.8 * ys + rng.standard_normal(50) * 0.2
        result = matched_variance_reduction(ys, yn)
        eff_n = result["effective_n_equivalent"]
        n = result["n_seeds"]
        unmatched_var = result["unmatched_variance"]
        matched_var = result["matched_variance"]
        assert eff_n == pytest.approx(n * unmatched_var / matched_var, rel=1e-6)

    def test_requires_1d_equal_length_arrays(self):
        with pytest.raises(ValueError, match="1-D arrays of equal length"):
            matched_variance_reduction([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_requires_at_least_two_seeds(self):
        with pytest.raises(ValueError, match="at least 2"):
            matched_variance_reduction([1.0], [2.0])

    def test_canonical_task_reduction_positive_for_positively_correlated_data(self):
        # Simulate matched seeds: hard/easy seeds show up in both conditions.
        rng = np.random.default_rng(12345)
        seed_difficulty = rng.standard_normal(8)          # shared across conditions
        ys = 10.0 + 2.0 * seed_difficulty + rng.standard_normal(8) * 0.5
        yn = 8.0 + 2.0 * seed_difficulty + rng.standard_normal(8) * 0.5
        result = matched_variance_reduction(ys, yn)
        assert result["variance_reduction_factor"] > 0.5
        assert result["within_seed_correlation"] > 0.5
        assert result["effective_n_equivalent"] > result["n_seeds"]


class TestPowerFromMatching:

    def test_power_increases_with_n_seeds(self):
        p_small = power_from_matching(8, mean_effect=1.0, matched_se_per_seed=2.0)
        p_large = power_from_matching(32, mean_effect=1.0, matched_se_per_seed=2.0)
        assert p_large > p_small

    def test_power_increases_with_effect_size(self):
        p1 = power_from_matching(16, mean_effect=1.0, matched_se_per_seed=2.0)
        p2 = power_from_matching(16, mean_effect=3.0, matched_se_per_seed=2.0)
        assert p2 > p1

    def test_high_power_for_large_effect(self):
        p = power_from_matching(100, mean_effect=5.0, matched_se_per_seed=1.0)
        assert p > 0.99

    def test_low_power_for_tiny_effect(self):
        p = power_from_matching(8, mean_effect=0.001, matched_se_per_seed=3.0)
        assert p < 0.10

    def test_invalid_se_raises(self):
        with pytest.raises(ValueError):
            power_from_matching(10, 1.0, matched_se_per_seed=-1.0)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            power_from_matching(0, 1.0, matched_se_per_seed=1.0)


# ---------------------------------------------------------------------------
# estimation.py
# ---------------------------------------------------------------------------

class TestApparentVelocityNoiseStd:

    def test_canonical_task_parameters(self):
        # σ=0.06, dt=0.02 → noise_std = 0.06/√0.02 ≈ 0.4243
        result = apparent_velocity_noise_std(diffusion_sigma=0.06, dt=0.02)
        assert result == pytest.approx(0.06 / np.sqrt(0.02), rel=1e-10)

    def test_larger_dt_reduces_noise(self):
        n1 = apparent_velocity_noise_std(0.06, dt=0.02)
        n2 = apparent_velocity_noise_std(0.06, dt=0.04)
        assert n2 < n1

    def test_larger_sigma_increases_noise(self):
        n1 = apparent_velocity_noise_std(0.06, dt=0.02)
        n2 = apparent_velocity_noise_std(0.12, dt=0.02)
        assert n2 == pytest.approx(2 * n1, rel=1e-10)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            apparent_velocity_noise_std(-0.06, 0.02)
        with pytest.raises(ValueError):
            apparent_velocity_noise_std(0.06, 0.0)


class TestFisherInformationFieldDirection:

    def test_fisher_proportional_to_n_valid(self):
        fi1 = fisher_information_field_direction(0.06, 0.4243, 10.0)
        fi2 = fisher_information_field_direction(0.06, 0.4243, 20.0)
        assert fi2["fisher_information_per_component"] == pytest.approx(
            2 * fi1["fisher_information_per_component"], rel=1e-10
        )

    def test_crb_decreases_with_more_observations(self):
        fi1 = fisher_information_field_direction(0.06, 0.4243, 5.0)
        fi2 = fisher_information_field_direction(0.06, 0.4243, 20.0)
        assert fi2["cramer_rao_bound_mse"] < fi1["cramer_rao_bound_mse"]

    def test_crb_times_fisher_equals_one(self):
        fi = fisher_information_field_direction(0.1, 0.3, 15.0)
        product = fi["fisher_information_per_component"] * fi["cramer_rao_bound_mse"]
        assert product == pytest.approx(1.0, rel=1e-10)

    def test_zero_observations_gives_infinite_crb(self):
        fi = fisher_information_field_direction(0.06, 0.4243, 0.0)
        assert fi["cramer_rao_bound_mse"] == float("inf")

    def test_invalid_noise_raises(self):
        with pytest.raises(ValueError):
            fisher_information_field_direction(0.06, 0.0, 10.0)

    def test_snr_formula(self):
        fi = fisher_information_field_direction(0.06, 0.4243, 10.0)
        expected_snr = 0.06 / 0.4243
        assert fi["signal_to_noise_ratio"] == pytest.approx(expected_snr, rel=1e-6)


class TestTeamSensingSummary:

    def test_sharing_gain_equals_n_collectors(self):
        # Team of M shares all M·K observations; gain in Fisher info is M.
        result = team_sensing_summary(
            signal_strength=0.06,
            diffusion_sigma=0.06,
            dt=0.02,
            n_collectors=4,
            k_particles_per_collector=32,
            validity_fraction=0.5,
        )
        assert result["sharing_fisher_gain"] == pytest.approx(4.0, rel=1e-10)

    def test_sharing_rmse_gain_equals_sqrt_n_collectors(self):
        result = team_sensing_summary(
            signal_strength=0.06,
            diffusion_sigma=0.06,
            dt=0.02,
            n_collectors=4,
            k_particles_per_collector=32,
            validity_fraction=0.5,
        )
        assert result["sharing_rmse_gain"] == pytest.approx(np.sqrt(4.0), rel=1e-6)

    def test_sufficiency_flag_always_true(self):
        result = team_sensing_summary(0.1, 0.06, 0.02, 2, 16, 0.3)
        assert result["sufficiency_holds"] is True

    def test_canonical_task_n_valid(self):
        result = team_sensing_summary(
            signal_strength=0.06,
            diffusion_sigma=0.06,
            dt=0.02,
            n_collectors=4,
            k_particles_per_collector=32,
            validity_fraction=0.5,
        )
        assert result["n_valid_per_agent"] == pytest.approx(16.0, rel=1e-10)
        assert result["n_valid_team_total"] == pytest.approx(64.0, rel=1e-10)

    def test_zero_validity_gives_zero_fisher(self):
        result = team_sensing_summary(0.06, 0.06, 0.02, 4, 32, validity_fraction=0.0)
        assert result["team_shared_fisher"] == pytest.approx(0.0, abs=1e-20)

    def test_invalid_validity_fraction_raises(self):
        with pytest.raises(ValueError):
            team_sensing_summary(0.06, 0.06, 0.02, 4, 32, validity_fraction=1.5)


# ---------------------------------------------------------------------------
# dilution.py
# ---------------------------------------------------------------------------

class TestOverlapProbability:

    def test_single_collector_no_overlap(self):
        p = overlap_probability(0.16, (2.0, 2.0), n_collectors=1)
        assert p == pytest.approx(0.0, abs=1e-10)

    def test_overlap_increases_with_collectors(self):
        p2 = overlap_probability(0.16, (2.0, 2.0), n_collectors=2)
        p4 = overlap_probability(0.16, (2.0, 2.0), n_collectors=4)
        assert p4 > p2

    def test_overlap_probability_nonnegative(self):
        for m in range(1, 8):
            p = overlap_probability(0.16, (2.0, 2.0), n_collectors=m)
            assert p >= 0.0


class TestExpectedUniqueCoverage:

    def test_single_collector_full_efficiency(self):
        result = expected_unique_coverage(0.16, (2.0, 2.0), n_collectors=1)
        assert result["sensing_budget_efficiency"] == pytest.approx(1.0, abs=1e-10)
        assert result["geometric_dilution_fraction"] == pytest.approx(0.0, abs=1e-10)

    def test_efficiency_decreases_with_more_collectors(self):
        r1 = expected_unique_coverage(0.16, (2.0, 2.0), n_collectors=2)
        r4 = expected_unique_coverage(0.16, (2.0, 2.0), n_collectors=4)
        assert r4["sensing_budget_efficiency"] < r1["sensing_budget_efficiency"]

    def test_canonical_task_geometry(self):
        # a/A = π·0.16²/4 ≈ 0.0201; M=4; geometric dilution should be small (<10%)
        result = expected_unique_coverage(0.16, (2.0, 2.0), n_collectors=4)
        assert result["coverage_fraction_per_collector"] == pytest.approx(
            np.pi * 0.16**2 / 4.0, rel=1e-6
        )
        assert result["geometric_dilution_fraction"] < 0.10

    def test_unique_coverage_fraction_bounded_by_one(self):
        result = expected_unique_coverage(0.16, (2.0, 2.0), n_collectors=4)
        assert 0.0 <= result["expected_unique_coverage_fraction"] <= 1.0


class TestSnrPerSeed:

    def test_shared_snr_is_sqrt_m_times_independent(self):
        M = 4
        shared = snr_per_seed(rho=2.0, n_collectors=M, k_particles_per_collector=32,
                              validity_fraction=0.5, shared=True)
        indep = snr_per_seed(rho=2.0, n_collectors=M, k_particles_per_collector=32,
                             validity_fraction=0.5, shared=False)
        assert shared["snr_per_step"] == pytest.approx(
            np.sqrt(M) * indep["snr_per_step"], rel=1e-10
        )

    def test_snr_proportional_to_rho(self):
        s1 = snr_per_seed(rho=1.0, n_collectors=4, k_particles_per_collector=32,
                          validity_fraction=0.5, shared=True)
        s2 = snr_per_seed(rho=2.0, n_collectors=4, k_particles_per_collector=32,
                          validity_fraction=0.5, shared=True)
        assert s2["snr_per_step"] == pytest.approx(2.0 * s1["snr_per_step"], rel=1e-10)

    def test_snr_zero_for_zero_rho(self):
        result = snr_per_seed(rho=0.0, n_collectors=4, k_particles_per_collector=32,
                              validity_fraction=0.5, shared=True)
        assert result["snr_per_step"] == pytest.approx(0.0, abs=1e-10)

    def test_canonical_task_alpha_006(self):
        # ρ = α·√dt/σ = 0.06·√0.02/0.06 = √0.02 ≈ 0.1414
        rho = 0.06 * np.sqrt(0.02) / 0.06
        result = snr_per_seed(rho=rho, n_collectors=4, k_particles_per_collector=32,
                              validity_fraction=0.5, shared=True)
        assert result["rho"] == pytest.approx(rho, rel=1e-10)
        # SNR > 0
        assert result["snr_per_step"] > 0.0


class TestDilutionSnrRatio:

    def test_ratio_equals_sqrt_m(self):
        for M in (2, 4, 8):
            result = dilution_snr_ratio(
                rho=1.0, n_collectors=M,
                k_particles_per_collector=32, validity_fraction=0.5,
            )
            assert result["snr_ratio_shared_over_independent"] == pytest.approx(
                np.sqrt(M), rel=1e-10
            )
            assert result["theoretical_sqrt_M_ratio"] == pytest.approx(np.sqrt(M), rel=1e-10)

    def test_canonical_task_m4_ratio_is_two(self):
        result = dilution_snr_ratio(
            rho=2.0, n_collectors=4,
            k_particles_per_collector=32, validity_fraction=0.5,
        )
        assert result["snr_ratio_shared_over_independent"] == pytest.approx(2.0, rel=1e-10)
