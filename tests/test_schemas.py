from __future__ import annotations

import json
import unittest
from pathlib import Path

from particle_benchmark.io import SchemaValidationError, validate_instance


ROOT = Path(__file__).resolve().parents[1]


class TestDatasetSchemas(unittest.TestCase):
    def test_numeric_minimum_maximum_and_exclusive_bounds_are_enforced(self) -> None:
        with self.assertRaisesRegex(SchemaValidationError, "above maximum"):
            validate_instance(1.1, {"type": "number", "maximum": 1.0})
        with self.assertRaisesRegex(SchemaValidationError, "exclusive minimum"):
            validate_instance(0.0, {"type": "number", "exclusiveMinimum": 0.0})
        with self.assertRaisesRegex(SchemaValidationError, "exclusive maximum"):
            validate_instance(1.0, {"type": "number", "exclusiveMaximum": 1.0})

    def test_aggregation_schemas_are_closed_and_truthful(self) -> None:
        diagnostic = json.loads(
            (ROOT / "schemas" / "message_diagnostic.schema.json").read_text()
        )
        self.assertFalse(diagnostic["additionalProperties"])
        for required in (
            "raw_summaries",
            "valid_particle_ids",
            "conditional_field_covariance",
            "analytic_ineligibility_reasons",
            "outcome_collector_positions",
            "action_implied_target_ids",
        ):
            self.assertIn(required, diagnostic["required"])
        manifest = json.loads(
            (ROOT / "schemas" / "aggregation_run_manifest.schema.json").read_text()
        )
        self.assertEqual(
            manifest["properties"]["pairing"]["properties"]["arms"]["prefixItems"],
            [{"const": "self_only"}, {"const": "all_to_all"}],
        )

    def test_trajectory_schema_is_parseable_and_closed(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "trajectory_step.schema.json").read_text()
        )
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("scenario_seed", schema["required"])
        self.assertIn("capture_events", schema["required"])

    def test_manifest_requires_reproducibility_provenance(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "run_manifest.schema.json").read_text()
        )
        self.assertFalse(schema["additionalProperties"])
        for field in ("repository", "environment", "pairing", "runtime", "artifacts"):
            self.assertIn(field, schema["required"])
        pairing = schema["properties"]["pairing"]["properties"]
        self.assertTrue(pairing["pre_generated_noise"]["const"])
        self.assertTrue(pairing["shared_initial_state"]["const"])


if __name__ == "__main__":
    unittest.main()
