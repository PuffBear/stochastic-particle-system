from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from particle_benchmark.io import (
    PreparedArtifact,
    SchemaValidationError,
    canonical_json_bytes,
    sha256_file,
    validate_instance,
    write_immutable_artifacts,
)


class TestArtifactIO(unittest.TestCase):
    def test_bounded_validator_rejects_missing_and_extra_keys(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {"value": {"type": "integer", "minimum": 0}},
        }
        validate_instance({"value": 3}, schema)
        with self.assertRaises(SchemaValidationError):
            validate_instance({}, schema)
        with self.assertRaises(SchemaValidationError):
            validate_instance({"value": 3, "hidden": 1}, schema)

    def test_canonical_json_is_sorted_and_rejects_nan(self) -> None:
        self.assertEqual(canonical_json_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}\n')
        with self.assertRaises(ValueError):
            canonical_json_bytes({"bad": float("nan")})

    def test_artifact_write_is_immutable_and_checksum_matches(self) -> None:
        artifact = PreparedArtifact("evidence.json", b'{"ok":true}\n', "application/json")
        with tempfile.TemporaryDirectory() as directory:
            (path,) = write_immutable_artifacts(directory, [artifact])
            self.assertEqual(path.read_bytes(), artifact.payload)
            self.assertEqual(sha256_file(path), artifact.sha256)
            with self.assertRaises(FileExistsError):
                write_immutable_artifacts(directory, [artifact])
            self.assertEqual(json.loads(path.read_text()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
