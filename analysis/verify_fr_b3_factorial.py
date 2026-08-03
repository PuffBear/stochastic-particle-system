#!/usr/bin/env python3
"""Verify a complete FR-B3 factorial before any confirmatory analysis."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
from typing import Mapping

import yaml

from particle_benchmark.io import canonical_json_bytes


EXPECTED_EXPERIMENT = "FR-B3-CATCHABILITY"
EXPECTED_STUDY = "factorial"
EXPECTED_BRANCH = "fr-b3-catchability-benchmark"
HEX_SHA = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob(root: Path, commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("episode_summaries.jsonl is empty")
    return rows


def validate_factorial_run(
    run_dir: Path,
    config_path: Path,
    *,
    repository_root: Path | None = None,
    verify_git_snapshot: bool = True,
) -> dict[str, object]:
    """Return a validation report; raise only for unreadable inputs."""

    failures: list[str] = []
    required = {
        "episode_summaries.jsonl",
        "design.json",
        "manifest.json",
    }
    missing = sorted(name for name in required if not (run_dir / name).is_file())
    if missing:
        return {"validation_passed": False, "failures": [f"missing files: {missing}"]}

    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    design = _load_json(run_dir / "design.json")
    manifest = _load_json(run_dir / "manifest.json")
    rows = _load_rows(run_dir / "episode_summaries.jsonl")

    if protocol.get("experiment_id") != EXPECTED_EXPERIMENT:
        failures.append("config experiment_id is not FR-B3-CATCHABILITY")
    if protocol.get("protocol_status") != "registered":
        failures.append("config protocol_status is not registered")
    if design.get("experiment_id") != EXPECTED_EXPERIMENT:
        failures.append("design experiment_id mismatch")
    if design.get("study") != EXPECTED_STUDY:
        failures.append("design is not the factorial study")
    if manifest.get("experiment_id") != EXPECTED_EXPERIMENT:
        failures.append("manifest experiment_id mismatch")
    if manifest.get("study") != EXPECTED_STUDY:
        failures.append("manifest is not the factorial study")
    if manifest.get("protocol_status") != "registered":
        failures.append("manifest protocol_status is not registered")
    if manifest.get("complete_frozen_design") is not True:
        failures.append("manifest does not mark a complete frozen design")
    limits = dict(design.get("development_limits", {}))
    if limits != {"max_cells": None, "max_seeds": None}:
        failures.append("design contains development limits")

    seeds = tuple(int(seed) for seed in protocol["seeds"])
    policies = tuple(str(policy) for policy in protocol["policies"])
    condition_ids = tuple(
        str(condition["condition_id"]) for condition in design.get("conditions", [])
    )
    if len(condition_ids) != len(set(condition_ids)):
        failures.append("design contains duplicate condition IDs")
    expected_cell_count = 1
    for values in dict(protocol["factorial_axes"]).values():
        expected_cell_count *= len(tuple(values))
    if len(condition_ids) != expected_cell_count:
        failures.append(
            f"condition count is {len(condition_ids)}, expected {expected_cell_count}"
        )
    expected_keys = set(itertools.product(condition_ids, seeds, policies))
    observed_keys: set[tuple[str, int, str]] = set()
    duplicate_keys: set[tuple[str, int, str]] = set()
    stream_groups: dict[tuple[str, int], set[str]] = {}
    condition_groups = {
        str(condition["condition_id"]): dict(condition["dimensionless_groups"])
        for condition in design.get("conditions", [])
    }
    particle_count = int(dict(protocol["fixed_environment"])["particle_count"])
    horizon = int(dict(protocol["fixed_environment"])["horizon"])

    for row in rows:
        key = (str(row.get("condition_id")), int(row.get("seed", -1)), str(row.get("policy_id")))
        if key in observed_keys:
            duplicate_keys.add(key)
        observed_keys.add(key)
        if row.get("study") != EXPECTED_STUDY:
            failures.append(f"non-factorial row at key {key}")
        value = int(row.get("unique_team_capture_yield", -1))
        if not 0 <= value <= particle_count:
            failures.append(f"capture yield outside [0, {particle_count}] at {key}")
        steps = int(row.get("executed_steps", -1))
        if not 1 <= steps <= horizon:
            failures.append(f"executed_steps outside [1, {horizon}] at {key}")
        expected_groups = condition_groups.get(key[0])
        if expected_groups is None:
            continue
        actual_groups = dict(row.get("dimensionless_groups", {}))
        for name in ("rho", "kappa", "eta"):
            if name not in actual_groups or abs(
                float(actual_groups[name]) - float(expected_groups[name])
            ) > 1e-12:
                failures.append(f"dimensionless group {name} mismatch at {key}")
        checksum_payload = json.dumps(
            row.get("stream_checksums", {}), sort_keys=True, separators=(",", ":")
        )
        stream_groups.setdefault((key[0], key[1]), set()).add(checksum_payload)

    if duplicate_keys:
        failures.append(f"duplicate episode keys: {len(duplicate_keys)}")
    missing_keys = expected_keys - observed_keys
    unexpected_keys = observed_keys - expected_keys
    if missing_keys:
        failures.append(f"missing episode keys: {len(missing_keys)}")
    if unexpected_keys:
        failures.append(f"unexpected episode keys: {len(unexpected_keys)}")
    mismatched_streams = [key for key, values in stream_groups.items() if len(values) != 1]
    if mismatched_streams:
        failures.append(f"policy stream mismatches: {len(mismatched_streams)}")

    expected_episode_count = len(expected_keys)
    if len(rows) != expected_episode_count:
        failures.append(f"row count is {len(rows)}, expected {expected_episode_count}")
    if int(design.get("episode_count", -1)) != expected_episode_count:
        failures.append("design episode_count mismatch")
    if int(design.get("condition_count", -1)) != len(condition_ids):
        failures.append("design condition_count mismatch")
    if int(design.get("seed_count", -1)) != len(seeds):
        failures.append("design seed_count mismatch")
    if tuple(str(item) for item in design.get("policies", [])) != policies:
        failures.append("design policy order differs from the registered config")

    for name, expected_hash in dict(manifest.get("artifacts", {})).items():
        artifact = run_dir / str(name)
        if not artifact.is_file():
            failures.append(f"manifest artifact is missing: {name}")
        elif _sha256(artifact) != str(expected_hash):
            failures.append(f"manifest artifact hash mismatch: {name}")

    repository = dict(manifest.get("repository", {}))
    commit = str(repository.get("commit_sha", ""))
    if repository.get("full_name") != "PuffBear/stochastic-particle-system":
        failures.append("manifest repository full_name mismatch")
    if repository.get("branch") != EXPECTED_BRANCH:
        failures.append("manifest branch mismatch")
    if not HEX_SHA.fullmatch(commit):
        failures.append("manifest commit is not a full lowercase SHA")
    elif verify_git_snapshot:
        root = repository_root or config_path.resolve().parents[2]
        for relative, expected_hash in dict(manifest.get("source_snapshot", {})).items():
            try:
                content = _git_blob(root, commit, str(relative))
            except (subprocess.CalledProcessError, FileNotFoundError):
                failures.append(f"cannot read source snapshot {relative} at {commit}")
                continue
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != str(expected_hash):
                failures.append(f"source snapshot hash mismatch: {relative}")

    return {
        "validation_passed": not failures,
        "failures": failures,
        "experiment_id": EXPECTED_EXPERIMENT,
        "study": EXPECTED_STUDY,
        "row_count": len(rows),
        "expected_row_count": expected_episode_count,
        "condition_count": len(condition_ids),
        "seed_count": len(seeds),
        "policy_count": len(policies),
        "paired_stream_group_count": len(stream_groups),
        "repository_commit": commit,
        "checks": [
            "registered protocol and complete-design flags",
            "exact condition-seed-policy Cartesian product",
            "unique episode grain",
            "capture and horizon domains",
            "dimensionless condition consistency",
            "common streams across policies",
            "artifact SHA-256 hashes",
            "Git source-snapshot SHA-256 hashes",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-git-snapshot", action="store_true")
    args = parser.parse_args()
    report = validate_factorial_run(
        args.run_dir,
        args.config,
        repository_root=args.repository_root,
        verify_git_snapshot=not args.skip_git_snapshot,
    )
    if args.report is not None:
        if args.report.exists():
            raise FileExistsError(f"immutable report already exists: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("xb") as handle:
            handle.write(canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
