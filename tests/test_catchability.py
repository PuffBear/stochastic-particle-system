"""FR-B3 design tests; these never execute registered scientific seeds."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from analysis.analyze_fr_b3_catchability import (
    leave_one_cell_out_rmse,
    paired_differences,
    validate_rescaling_audit,
)
from analysis.calibrate_fr_b3_design import loco_rmses
from analysis.run_fr_b3_catchability import (
    _canonicalize_policy_observations,
    factorial_conditions,
    load_protocol,
)
from particle_benchmark.catchability import (
    catchability_groups,
    physical_parameters_from_groups,
    rescale_equivalent_config,
)
from particle_benchmark.environment import ParticleEnvConfig


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "experiments" / "fr_b3_catchability.yaml"


class DimensionlessCatchabilityTests(unittest.TestCase):
    def test_audit_velocity_canonicalization_is_scale_invariant(self) -> None:
        particles = np.zeros((2, 5), dtype=np.float64)
        particles[:, 2:4] = [[1.25, -0.75], [0.5, 2.0]]
        observation = {
            "particles": particles,
            "particle_mask": np.asarray([True, True]),
            "velocity_valid_mask": np.asarray([True, True]),
        }
        scaled = {**observation, "particles": particles * [1.0, 1.0, 2.0, 2.0, 1.0]}
        converted = _canonicalize_policy_observations((scaled,), velocity_scale=0.5)
        np.testing.assert_array_equal(converted[0]["particles"], particles)
        np.testing.assert_array_equal(converted[0]["particle_mask"], observation["particle_mask"])

    def test_executed_c03_anchor_is_kappa_point_five(self) -> None:
        config = ParticleEnvConfig(
            horizon=67,
            dt=0.02,
            signal_strength=0.06,
            diffusion_sigma=0.06,
            collector_max_speed=0.12,
        )
        groups = catchability_groups(config)
        self.assertAlmostEqual(groups.rho, math.sqrt(0.02))
        self.assertAlmostEqual(groups.kappa, 0.5)
        self.assertAlmostEqual(groups.eta, 0.06 * math.sqrt(0.02))
        self.assertAlmostEqual(
            groups.normalized_drift_step, groups.rho * groups.eta
        )
        self.assertAlmostEqual(
            groups.normalized_control_step,
            groups.rho * groups.eta / groups.kappa,
        )

    def test_group_to_physical_round_trip(self) -> None:
        physical = physical_parameters_from_groups(
            rho=math.sqrt(0.02),
            kappa=0.5,
            eta=0.06 * math.sqrt(0.02),
            dt=0.02,
            arena_size=(1.0, 1.0),
        )
        self.assertAlmostEqual(physical["signal_strength"], 0.06)
        self.assertAlmostEqual(physical["diffusion_sigma"], 0.06)
        self.assertAlmostEqual(physical["collector_max_speed"], 0.12)

    def test_equivalent_rescaling_preserves_every_frozen_group(self) -> None:
        base = ParticleEnvConfig(
            horizon=67,
            dt=0.02,
            signal_strength=0.06,
            diffusion_sigma=0.06,
            collector_max_speed=0.12,
        )
        expected = catchability_groups(base)
        for length_scale, time_scale in ((2.0, 1.0), (1.0, 4.0), (0.5, 0.25)):
            actual = catchability_groups(
                rescale_equivalent_config(
                    base, length_scale=length_scale, time_scale=time_scale
                )
            )
            np.testing.assert_allclose(
                list(actual.to_dict().values()),
                list(expected.to_dict().values()),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_protocol_is_full_three_by_three_by_three_factorial(self) -> None:
        protocol = load_protocol(PROTOCOL)
        conditions = factorial_conditions(protocol)
        self.assertEqual(len(conditions), 27)
        self.assertEqual(len(protocol["seeds"]), 64)
        anchor = conditions[13]
        groups = catchability_groups(anchor.environment)
        self.assertAlmostEqual(groups.rho, math.sqrt(0.02))
        self.assertAlmostEqual(groups.kappa, 0.5)
        self.assertAlmostEqual(anchor.environment.collector_max_speed, 0.12)


class CatchabilityAnalysisTests(unittest.TestCase):
    def test_calibration_loco_operator_rewards_true_eta_structure(self) -> None:
        coordinates = np.asarray(
            [
                (x, y, z)
                for x in (-1.0, 0.0, 1.0)
                for y in (-1.0, 0.0, 1.0)
                for z in (-1.0, 0.0, 1.0)
            ]
        )
        outcomes = (1.0 + coordinates[:, 0] + coordinates[:, 2])[:, None]
        two_axis, three_axis = loco_rmses(outcomes)
        self.assertGreater(float(two_axis[0]), 0.5)
        self.assertLess(float(three_axis[0]), 1e-10)

    def test_pairing_rejects_a_missing_primary_arm(self) -> None:
        rows = [
            {
                "study": "factorial",
                "condition_id": "r0_k0_e0",
                "seed": 17,
                "policy_id": "shared_summary_v2",
                "unique_team_capture_yield": 3,
            }
        ]
        with self.assertRaisesRegex(ValueError, "missing primary paired arm"):
            paired_differences(rows)

    def test_three_axis_features_predict_eta_dependent_surface_better(self) -> None:
        coordinates = [
            (rho, kappa, eta)
            for rho in (0.5, 1.0, 2.0)
            for kappa in (0.5, 1.0, 2.0)
            for eta in (0.5, 1.0, 2.0)
        ]
        outcomes = np.asarray(
            [
                1.0
                + math.log2(rho) ** 2
                - 0.5 * math.log2(kappa)
                + 3.0 * math.log2(eta)
                for rho, kappa, eta in coordinates
            ]
        )
        two_axis_rmse, _ = leave_one_cell_out_rmse(
            coordinates, outcomes, anchor=(1.0, 1.0, 1.0), include_eta=False
        )
        three_axis_rmse, _ = leave_one_cell_out_rmse(
            coordinates, outcomes, anchor=(1.0, 1.0, 1.0), include_eta=True
        )
        self.assertGreater(two_axis_rmse, 1.0)
        self.assertLess(three_axis_rmse, 1e-10)

    def test_rescaling_audit_requires_yield_and_state_identity(self) -> None:
        base = {
            "study": "rescaling_audit",
            "seed": 17,
            "policy_id": "shared_summary_v2",
            "unique_team_capture_yield": 8,
            "normalized_final_state_sha256": "same",
        }
        passed = validate_rescaling_audit(
            [{**base, "condition_id": "canonical"}, {**base, "condition_id": "scaled"}]
        )
        self.assertTrue(passed["audit_passed"])
        failed = validate_rescaling_audit(
            [
                {**base, "condition_id": "canonical"},
                {
                    **base,
                    "condition_id": "scaled",
                    "normalized_final_state_sha256": "different",
                },
            ]
        )
        self.assertFalse(failed["audit_passed"])


if __name__ == "__main__":
    unittest.main()
