from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from particle_benchmark.environment import ParticleEnvConfig
from particle_benchmark.io import load_schema, validate_instance
from particle_benchmark.runner import (
    PairExperimentConfig,
    RepositoryProvenance,
    load_pair_config,
    run_config_file,
    run_matched_pair,
    validate_paired_environment_configs,
)


ROOT = Path(__file__).resolve().parents[1]


class TestMatchedRunner(unittest.TestCase):
    def _environment(self) -> ParticleEnvConfig:
        return ParticleEnvConfig(
            particle_count=10,
            collector_count=2,
            horizon=4,
            dt=0.02,
            diffusion_sigma=0.03,
            collector_radius=0.01,
            nearest_particles_k=4,
            field_family="uniform",
            field_kwargs={"orientation": 0.3},
            signal_strength=0.0,
        )

    def _config(self, **overrides: object) -> PairExperimentConfig:
        values: dict[str, object] = {
            "experiment_id": "SPS-PAIR-TEST",
            "scenario_seed": 17,
            "signal_strength": 0.0,
            "policy_id": "pregenerated_random",
            "environment": self._environment(),
            "repository": RepositoryProvenance(
                full_name="PuffBear/stochastic-particle-system",
                branch="research-autonomy",
                commit_sha="a" * 40,
                dirty=False,
            ),
            "created_at_utc": "2026-07-31T09:00:00Z",
            "config_path": "tests/in-memory-pair-config.yaml",
            "run_command": "python -m test_runner",
        }
        values.update(overrides)
        return PairExperimentConfig(**values)

    def test_zero_signal_full_pair_is_identical_and_schema_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_matched_pair(self._config(), directory)
            self.assertTrue(result.summary["zero_signal_identity_checked"])
            self.assertEqual(result.summary["matched_first_interception_gain"], 0.0)
            self.assertEqual(
                result.paths.null_trajectory.read_bytes(),
                result.paths.signal_trajectory.read_bytes().replace(
                    b'"condition":"signal"', b'"condition":"null"'
                ),
            )
            manifest = json.loads(result.paths.manifest.read_text())
            validate_instance(
                manifest, load_schema(ROOT / "schemas" / "run_manifest.schema.json")
            )
            self.assertTrue(manifest["pairing"]["difference_only_signal_strength"])
            self.assertEqual(len(manifest["artifacts"]), 3)
            for artifact in manifest["artifacts"]:
                self.assertEqual(len(artifact["sha256"]), 64)

    def test_nonzero_pair_preserves_all_input_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_matched_pair(
                self._config(signal_strength=0.25, policy_id="local_flow_v1"),
                directory,
            )
            checksums = result.summary["input_checksums"]
            self.assertEqual(set(checksums), {
                "initial_state_sha256",
                "brownian_sha256",
                "field_nuisance_sha256",
                "policy_randomness_sha256",
                "tie_key_provenance_sha256",
            })
            self.assertFalse(result.summary["zero_signal_identity_checked"])

    def test_precontact_endpoint_stops_each_arm_after_first_interception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_matched_pair(
                self._config(
                    signal_strength=2.0,
                    policy_id="stationary",
                    stop_after_first_interception=True,
                ),
                directory,
            )
            self.assertTrue(result.summary["stopped_after_first_interception"])
            for condition in ("null", "signal"):
                summary = result.summary[condition]
                if not summary["censored"]:
                    self.assertEqual(
                        summary["trajectory_steps"],
                        summary["first_interception_step"],
                    )

    def test_hidden_pair_difference_is_rejected(self) -> None:
        null = replace(self._environment(), signal_strength=0.0)
        signal = replace(null, signal_strength=0.2, dt=0.1)
        with self.assertRaisesRegex(ValueError, "dt"):
            validate_paired_environment_configs(null, signal)

    def test_run_id_collision_cannot_overwrite_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(policy_id="stationary")
            run_matched_pair(config, directory)
            with self.assertRaises(FileExistsError):
                run_matched_pair(config, directory)

    def test_strict_yaml_config_runner(self) -> None:
        config = self._config(policy_id="coverage", policy_config={"sweep_period": 5})
        raw = {
            "experiment_id": config.experiment_id,
            "scenario_seed": config.scenario_seed,
            "signal_strength": config.signal_strength,
            "policy_id": config.policy_id,
            "policy_config": config.policy_config,
            "environment": {
                **config.environment.__dict__,
                "arena_size": list(config.environment.arena_size),
            },
            "repository": config.repository.__dict__,
            "created_at_utc": config.created_at_utc,
            "run_command": config.run_command,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=True))
            loaded = load_pair_config(path)
            self.assertEqual(loaded.policy_id, "coverage")
            result = run_config_file(path, Path(directory) / "outputs")
            self.assertEqual(result.summary["policy_id"], "coverage")


if __name__ == "__main__":
    unittest.main()
