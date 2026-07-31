"""Immutable, schema-checked JSON artifacts for benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class SchemaValidationError(ValueError):
    """Raised when an artifact does not satisfy its checked JSON schema."""


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    """Encode JSON deterministically and reject NaN/infinity."""
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + suffix
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    """Hash a canonical JSON value (without a trailing newline)."""
    return sha256_bytes(canonical_json_bytes(value, newline=False))


def _json_type_matches(instance: object, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return (
            isinstance(instance, (int, float))
            and not isinstance(instance, bool)
            and math.isfinite(float(instance))
        )
    if expected == "string":
        return isinstance(instance, str)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "object":
        return isinstance(instance, dict)
    raise SchemaValidationError(f"unsupported schema type {expected!r}")


def _resolve_local_ref(root_schema: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise SchemaValidationError(f"only local schema references are supported: {reference}")
    node: Any = root_schema
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or token not in node:
            raise SchemaValidationError(f"unresolved schema reference: {reference}")
        node = node[token]
    if not isinstance(node, Mapping):
        raise SchemaValidationError(f"schema reference is not an object: {reference}")
    return node


def _validate(instance: object, schema: Mapping[str, Any], root: Mapping[str, Any], path: str) -> None:
    """Validate the deliberately small JSON-Schema subset used by this repo."""
    if "$ref" in schema:
        _validate(instance, _resolve_local_ref(root, str(schema["$ref"])), root, path)
        return
    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"{path}: value is not in enum")
    if "oneOf" in schema:
        successes = 0
        for option in schema["oneOf"]:
            try:
                _validate(instance, option, root, path)
                successes += 1
            except SchemaValidationError:
                pass
        if successes != 1:
            raise SchemaValidationError(f"{path}: expected exactly one matching oneOf branch")
        return

    expected = schema.get("type")
    if expected is not None:
        expected_types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_json_type_matches(instance, item) for item in expected_types):
            raise SchemaValidationError(f"{path}: expected type {expected_types}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise SchemaValidationError(f"{path}: missing required keys {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(instance) - set(properties))
            if extras:
                raise SchemaValidationError(f"{path}: unexpected keys {extras}")
        for key, value in instance.items():
            child_schema = properties.get(key)
            if child_schema is not None:
                _validate(value, child_schema, root, f"{path}.{key}")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < int(schema["minItems"]):
            raise SchemaValidationError(f"{path}: too few items")
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            raise SchemaValidationError(f"{path}: too many items")
        prefix = schema.get("prefixItems", [])
        for index, child_schema in enumerate(prefix[: len(instance)]):
            _validate(instance[index], child_schema, root, f"{path}[{index}]")
        items = schema.get("items")
        if items is False and len(instance) > len(prefix):
            raise SchemaValidationError(f"{path}: unexpected trailing items")
        if isinstance(items, Mapping):
            for index in range(len(prefix), len(instance)):
                _validate(instance[index], items, root, f"{path}[{index}]")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            raise SchemaValidationError(f"{path}: string is too short")
        if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
            raise SchemaValidationError(f"{path}: string does not match pattern")
        if schema.get("format") == "date-time":
            try:
                datetime.fromisoformat(instance.replace("Z", "+00:00"))
            except ValueError as exc:
                raise SchemaValidationError(f"{path}: invalid date-time") from exc

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and float(instance) < float(schema["minimum"]):
            raise SchemaValidationError(f"{path}: below minimum")


def validate_instance(instance: object, schema: Mapping[str, Any]) -> None:
    """Validate an instance against the repository's bounded schema subset."""
    _validate(instance, schema, schema, "$")


def load_schema(path: str | Path) -> dict[str, Any]:
    """Load a JSON schema object."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SchemaValidationError("schema root must be an object")
    return value


def validate_jsonl(records: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> None:
    """Validate every JSONL record before any bytes are written."""
    for index, record in enumerate(records):
        try:
            validate_instance(record, schema)
        except SchemaValidationError as exc:
            raise SchemaValidationError(f"record {index}: {exc}") from exc


@dataclass(frozen=True)
class PreparedArtifact:
    """A validated artifact ready for an immutable write."""

    filename: str
    payload: bytes
    media_type: str

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.payload)


def prepare_json(filename: str, value: Mapping[str, Any], schema: Mapping[str, Any]) -> PreparedArtifact:
    validate_instance(value, schema)
    return PreparedArtifact(filename, canonical_json_bytes(value), "application/json")


def prepare_jsonl(
    filename: str,
    records: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
) -> PreparedArtifact:
    validate_jsonl(records, schema)
    payload = b"".join(canonical_json_bytes(record) for record in records)
    return PreparedArtifact(filename, payload, "application/x-ndjson")


def write_immutable_artifacts(
    output_dir: str | Path, artifacts: Sequence[PreparedArtifact]
) -> tuple[Path, ...]:
    """Create all artifacts without overwriting any existing evidence file."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    names = [artifact.filename for artifact in artifacts]
    if len(names) != len(set(names)):
        raise ValueError("artifact filenames must be unique")
    paths = tuple(directory / name for name in names)
    collisions = [path for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(f"immutable artifact already exists: {collisions[0]}")

    written: list[Path] = []
    try:
        for path, artifact in zip(paths, artifacts, strict=True):
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(artifact.payload)
                handle.flush()
                os.fsync(handle.fileno())
            written.append(path)
    except Exception:
        # Completed evidence remains immutable. A partial bundle is explicit and
        # must be inspected rather than silently removed or overwritten.
        raise
    return tuple(written)
