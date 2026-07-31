from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestDatasetSchemas(unittest.TestCase):
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
