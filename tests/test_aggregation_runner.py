from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from particle_benchmark.aggregation_runner import (
    AggregationPairExperimentConfig,
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
                "unresolved_T_1.34_vs_canonical_8.0_no_scientific_endpoint_selected",
            )
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


if __name__ == "__main__":
    unittest.main()
