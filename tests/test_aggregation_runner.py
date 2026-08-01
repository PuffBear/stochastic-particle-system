from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from particle_benchmark.aggregation_runner import (
    AggregationPairExperimentConfig,
    _arm_summary,
    evaluate_arm_eta_diagnostic_gate,
    load_aggregation_pair_config,
    run_aggregation_pair,
)
from particle_benchmark.environment import ParticleEnvConfig
from particle_benchmark.io import load_schema, validate_instance
from particle_benchmark.runner import RepositoryProvenance


ROOT = Path(__file__).resolve().parents[1]


class TestAggregationRunner(unittest.TestCase):
    def _environment(self) -> ParticleEnvConfig:
        return ParticleEnvConfig(
            particle_count=4,
            collector_count=4,
            horizon=2,
            dt=0.01,
            diffusion_sigma=0.0,
            collector_max_speed=0.05,
            sensing_radius=2.0,
            collector_radius=0.0,
            particle_radius=0.0,
            nearest_particles_k=2,
            field_family="periodic_gaussian",
            signal_strength=0.01,
            field_kwargs={
                "correlation_length": 0.2,
                "component_variance": 0.01,
                "max_frequency": 3,
            },
            include_particle_velocity=True,
            include_teammates=False,
        )

    def _config(self, *, identity: bool = False) -> AggregationPairExperimentConfig:
        return AggregationPairExperimentConfig(
            experiment_id=("SPS-WO-10-IDENTITY" if identity else "SPS-WO-10-PAIR"),
            scenario_seed=9100,
            environment=self._environment(),
            repository=RepositoryProvenance(
                full_name="PuffBear/stochastic-particle-system",
                branch="research-autonomy",
                commit_sha="d4a497ab50021d2ed17289d1ba56cf420075947a",
                dirty=False,
            ),
            created_at_utc="2026-08-01T00:00:00Z",
            config_path="tests/in-memory-aggregation.yaml",
            run_command="deterministic unittest only",
            arm_modes=(
                {"self_only": "independent", "all_to_all": "independent"}
                if identity
                else {"self_only": "independent", "all_to_all": "all_to_all"}
            ),
        )

    def test_canonical_pair_is_stream_matched_schema_valid_and_collision_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_aggregation_pair(self._config(), directory)
            self.assertEqual(result.summary["equal_source_state_checks"], 2)
            self.assertEqual(result.summary["equal_sent_message_checks"], 2)
            self.assertFalse(result.summary["zero_intervention_identity_checked"])
            self.assertEqual(
                result.summary["endpoint_status"],
                "frozen_T_1.34_current_run_nonendpoint_diagnostic",
            )
            self.assertEqual(result.summary["frozen_physical_endpoint_seconds"], 1.34)
            manifest = json.loads(result.paths.manifest.read_text())
            validate_instance(
                manifest,
                load_schema(ROOT / "schemas" / "aggregation_run_manifest.schema.json"),
            )
            rows = [
                json.loads(line)
                for line in result.paths.all_to_all_diagnostics.read_text().splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertIn("initial_no_history", rows[0]["analytic_ineligibility_reasons"])
            self.assertTrue(rows[1]["analytic_eligible"])
            self.assertIsNotNone(rows[1]["conditional_field_covariance"])
            self.assertEqual(rows[1]["outcome_step"], rows[1]["decision_step"] + 1)
            post = result.summary["all_to_all"]["post_history_diagnostics"]
            self.assertEqual(post["rows"], 1)
            self.assertEqual(post["analytic_eligibility"], {
                "numerator": 1,
                "denominator": 1,
                "rate": 1.0,
            })
            self.assertEqual(post["action_transitions"], 4)
            with self.assertRaises(FileExistsError):
                run_aggregation_pair(self._config(), directory)

    def test_same_mode_identity_control_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_aggregation_pair(self._config(identity=True), directory)
            self.assertTrue(result.summary["zero_intervention_identity_checked"])
            left = result.paths.self_only_diagnostics.read_text().replace(
                '"arm":"self_only"', '"arm":"same"'
            )
            right = result.paths.all_to_all_diagnostics.read_text().replace(
                '"arm":"all_to_all"', '"arm":"same"'
            )
            self.assertEqual(left, right)

    def test_frozen_config_fixture_loads_without_execution(self) -> None:
        config = load_aggregation_pair_config(
            ROOT / "configs" / "experiments" / "sps_wo10_deterministic_integration.yaml"
        )
        self.assertEqual(config.environment.field_family, "periodic_gaussian")
        self.assertFalse(config.environment.include_teammates)
        self.assertEqual(config.arm_modes["self_only"], "independent")

    @staticmethod
    def _synthetic_row(
        decision_step: int,
        *,
        eligible: bool,
        captures: int = 0,
        rescue: tuple[bool, bool, bool, bool] = (False, False, False, False),
        cancellation: tuple[bool, bool, bool, bool] = (False, False, False, False),
        own_empty: tuple[bool, bool, bool, bool] = (False, False, False, False),
        outcome_reflection: tuple[bool, bool, bool, bool] = (False, False, False, False),
    ) -> dict[str, object]:
        return {
            "decision_step": decision_step,
            "unique_yield_so_far": captures,
            "transition_unique_captures": captures,
            "valid_particle_counts": [2, 2, 2, 2],
            "clipped_components": [[False, False] for _ in range(4)],
            "reflected_valid_counts": [0, 0, 0, 0],
            "outcome_particle_reflection_count": 0,
            "own_empty": list(own_empty),
            "fallback_used": [False, False, False, False],
            "cross_agent_rescue": list(rescue),
            "zero_direction_cancellation": list(cancellation),
            "outcome_collector_reflected": list(outcome_reflection),
            "analytic_eligible": eligible,
        }

    def test_post_history_counts_keep_treatment_events_and_exclude_only_reflected_mediator_units(self) -> None:
        records = [
            self._synthetic_row(0, eligible=False, captures=1),
            self._synthetic_row(
                1,
                eligible=False,
                captures=2,
                rescue=(True, False, False, False),
                own_empty=(True, False, False, False),
            ),
            self._synthetic_row(
                2,
                eligible=True,
                captures=3,
                cancellation=(False, True, False, False),
                outcome_reflection=(False, False, True, False),
            ),
        ]
        summary = _arm_summary(records, nearest_particles_k=2)
        self.assertEqual(summary["steps"], 3)
        self.assertEqual(summary["captured_total"], 3)
        post = summary["post_history_diagnostics"]
        self.assertEqual(post["rows"], 2)
        self.assertEqual(post["analytic_eligibility"]["rate"], 0.5)
        self.assertEqual(
            post["cross_agent_rescue_own_empty_rows"],
            {"numerator": 1, "denominator": 1, "rate": 1.0},
        )
        self.assertEqual(
            post["zero_direction_cancellation_nonfallback_agent_rows"],
            {"numerator": 1, "denominator": 8, "rate": 0.125},
        )
        self.assertEqual(
            post["collector_reflected_action_transitions"],
            {"numerator": 1, "denominator": 8, "rate": 0.125},
        )
        self.assertEqual(
            post["action_displacement_eligible_transitions"],
            {"numerator": 7, "denominator": 8, "rate": 0.875},
        )

    def test_frozen_gate_thresholds_are_inclusive_and_rescue_is_uncapped(self) -> None:
        rates = {
            arm: {
                "analytic_eligibility_rate": 0.80,
                "clipped_velocity_component_rate": 0.01,
                "own_empty_agent_row_rate": 0.05,
                "fallback_agent_row_rate": 0.05,
                "reflected_valid_slot_rate": 0.05,
                "collector_reflected_action_transition_rate": 0.05,
                "cross_agent_rescue_rate": 1.0,
                "zero_direction_cancellation_rate": 1.0,
            }
            for arm in ("self_only", "all_to_all")
        }
        rates["self_only"]["analytic_eligibility_rate"] = 0.90
        report = evaluate_arm_eta_diagnostic_gate(
            rates,
            {"self_only": [0.50], "all_to_all": [0.50]},
        )
        self.assertTrue(report["gate_passed"])
        self.assertAlmostEqual(
            report["analytic_eligibility_absolute_rate_difference"], 0.10
        )
        self.assertAlmostEqual(
            report["analytic_eligibility_arm_gap_percentage_points"], 10.0
        )

        rates["self_only"]["analytic_eligibility_rate"] = 0.9001
        report = evaluate_arm_eta_diagnostic_gate(
            rates,
            {"self_only": [0.50], "all_to_all": [0.50]},
        )
        self.assertFalse(report["gate_passed"])
        self.assertFalse(report["components"]["analytic_eligibility_arm_gap"])


if __name__ == "__main__":
    unittest.main()
