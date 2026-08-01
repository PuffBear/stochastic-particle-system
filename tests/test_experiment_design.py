"""Deterministic design tests; these never execute registered scientific seeds."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
import yaml

from analysis import run_attribution_controls as attribution
from analysis import run_power_analysis as power
from analysis import run_timestep_convergence as timestep


class CoupledTimestepTests(unittest.TestCase):
    def test_fixed_physical_duration(self) -> None:
        self.assertEqual(timestep._evaluation_steps(0.02), 67)
        self.assertEqual(timestep._evaluation_steps(0.01), 134)
        self.assertEqual(timestep._evaluation_steps(0.005), 268)

    def test_coarse_increments_equal_summed_finest_increments(self) -> None:
        bank = timestep.coupled_noise_bank(17, particle_count=3)
        finest_increments = bank[0.005] * math.sqrt(0.005)
        for dt, factor in ((0.01, 2), (0.02, 4)):
            expected = finest_increments.reshape(268 // factor, factor, 3, 2).sum(axis=1)
            actual = bank[dt] * math.sqrt(dt)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-15)

    @staticmethod
    def _summaries(*, dt01_oracle_gain: int = 2) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for seed in timestep.FROZEN_SEEDS:
            for dt in timestep.FROZEN_DT_VALUES:
                checks = {
                    "initial_state_sha256": f"init-{seed}",
                    "field_nuisance_sha256": f"field-{seed}",
                    "coupled_finest_brownian_sha256": f"fine-{seed}",
                    "brownian_sha256": f"brownian-{seed}-{dt}",
                }
                for alpha in timestep.FROZEN_ALPHAS:
                    for policy in timestep.FROZEN_POLICIES:
                        stationary = 4 if alpha == 0.0 else 5
                        gain = dt01_oracle_gain if dt == 0.01 else 2
                        yield_value = stationary if policy == "stationary" else stationary + gain
                        rows.append(
                            {
                                "seed": seed,
                                "alpha": alpha,
                                "policy_id": policy,
                                "dt": dt,
                                "unique_team_capture_yield": yield_value,
                                "executed_steps": timestep._evaluation_steps(dt),
                                "stream_checksums": checks,
                            }
                        )
        return rows

    def test_gate_matches_preregistration(self) -> None:
        report = timestep.evaluate_gate(self._summaries())
        self.assertTrue(report["convergence_gate_passed"])
        self.assertEqual(
            report["pairwise_differences"]["dt02_to_dt01_seed_sign_change_count"], 0
        )

    def test_strict_mean_tolerance_can_fail(self) -> None:
        report = timestep.evaluate_gate(self._summaries(dt01_oracle_gain=3))
        self.assertFalse(report["convergence_gate_passed"])


class AttributionGateTests(unittest.TestCase):
    @staticmethod
    def _summaries(
        *, alpha0_shared_advantage: int = 0, favorable_signal_gain: int = 4
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, seed in enumerate(attribution.FROZEN_SEEDS):
            checks = {
                "initial_state_sha256": f"initial-{seed}",
                "brownian_sha256": f"brownian-{seed}",
                "field_nuisance_sha256": f"field-{seed}",
                "tie_key_provenance_sha256": f"tie-{seed}",
            }
            for alpha in attribution.FROZEN_ALPHAS:
                for policy in attribution.FROZEN_POLICIES:
                    if alpha == 0.0:
                        value = {
                            "stationary": 4,
                            "capacity_matched_independent": 6,
                            "shared_summary": 6 + alpha0_shared_advantage,
                            "full_state_interception_oracle": 10,
                        }[policy]
                    else:
                        value = {
                            "stationary": 4,
                            "capacity_matched_independent": 5,
                            "shared_summary": 5 + favorable_signal_gain if index < 5 else 5,
                            "full_state_interception_oracle": 10,
                        }[policy]
                    rows.append(
                        {
                            "seed": seed,
                            "alpha": alpha,
                            "policy_id": policy,
                            "unique_team_capture_yield": value,
                            "executed_steps": attribution.EVALUATION_STEPS,
                            "stream_checksums": checks,
                        }
                    )
        return rows

    def test_policy_ids_are_runner_supported(self) -> None:
        self.assertEqual(
            attribution.FROZEN_POLICIES,
            (
                "capacity_matched_independent",
                "shared_summary",
                "stationary",
                "full_state_interception_oracle",
            ),
        )

    def test_joint_gate_matches_frozen_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        frozen = yaml.safe_load(
            (root / "configs/experiments/sps_wo07_attribution_controls.yaml").read_text()
        )
        self.assertEqual(
            frozen["joint_gate"],
            {
                "minimum_mean_shared_minus_independent_particles": 2.0,
                "minimum_positive_shared_minus_independent_seeds": 5,
                "mean_shared_minus_stationary_strictly_above": 0.0,
                "mean_signal_difference_in_differences_strictly_above": 0.0,
                "require_correctness_and_execution": True,
                "require_matched_streams": True,
            },
        )

    def test_all_preregistered_gate_components_are_enforced(self) -> None:
        report = attribution.evaluate_gate(self._summaries())
        self.assertTrue(report["attribution_gate_passed"])
        self.assertTrue(all(report["gate_components"].values()))

    def test_raw_contrast_cannot_rescue_failed_signal_difference_in_differences(self) -> None:
        report = attribution.evaluate_gate(self._summaries(alpha0_shared_advantage=3))
        self.assertGreater(report["primary_contrast"]["mean"], 0.0)
        self.assertFalse(
            report["gate_components"]["mean_signal_difference_in_differences_positive"]
        )
        self.assertFalse(report["attribution_gate_passed"])
        self.assertFalse(report["full_confirmation_run_warranted"])

    def test_substantively_small_mean_fails_relevance_gate(self) -> None:
        report = attribution.evaluate_gate(self._summaries(favorable_signal_gain=3))
        self.assertGreater(report["primary_contrast"]["mean"], 0.0)
        self.assertLess(report["primary_contrast"]["mean"], 2.0)
        self.assertFalse(
            report["gate_components"]["mean_shared_minus_independent_at_least_2_particles"]
        )
        self.assertFalse(report["attribution_gate_passed"])

    def test_matched_stream_failure_is_invalid(self) -> None:
        rows = self._summaries()
        rows[1] = {
            **rows[1],
            "stream_checksums": {
                **dict(rows[1]["stream_checksums"]),
                "brownian_sha256": "mismatch",
            },
        }
        report = attribution.evaluate_gate(rows)
        self.assertFalse(report["gate_components"]["matched_streams_verified"])
        self.assertEqual(
            report["interpretation"], "invalid_correctness_or_provenance_failure"
        )
        self.assertFalse(report["attribution_gate_passed"])


class PowerDesignTests(unittest.TestCase):
    def test_power_report_is_ex_ante_one_sided_and_paired(self) -> None:
        report = power.build_power_report(
            assumed_sd_values=np.asarray([2.0, 4.0]),
            seed_counts=(8, 12),
            bootstrap_draws=100,
            calibration_datasets=3,
            simulation_trials=100,
            rng_seed=123,
        )
        self.assertEqual(report["coordination_minimum_effect"], 2.0)
        self.assertTrue(report["minimum_effect_frozen_ex_ante"])
        self.assertIn("one-sided paired", report["test"])
        self.assertEqual(report["seed_counts_evaluated"], [8, 12])


if __name__ == "__main__":
    unittest.main()
