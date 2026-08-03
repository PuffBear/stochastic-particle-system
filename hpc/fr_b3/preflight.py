#!/usr/bin/env python3
"""Fail closed before consuming any registered FR-B3 factorial seed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


EXPECTED_BRANCH = "fr-b3-catchability-benchmark"
EXPECTED_EPISODES = 6912


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    branch = _git(root, "branch", "--show-current")
    commit = _git(root, "rev-parse", "HEAD")
    tracked_status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if branch != EXPECTED_BRANCH:
        raise SystemExit(f"wrong branch: {branch!r}; expected {EXPECTED_BRANCH!r}")
    if len(commit) != 40:
        raise SystemExit("HEAD is not a full commit SHA")
    if tracked_status:
        raise SystemExit("tracked worktree changes are present; refuse frozen run")
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if protocol.get("experiment_id") != "FR-B3-CATCHABILITY":
        raise SystemExit("wrong experiment config")
    if protocol.get("protocol_status") != "registered":
        raise SystemExit("FR-B3 protocol is not registered")
    for path in (
        args.run_output,
        args.validation_output,
        args.analysis_output,
        args.figure_output,
    ):
        if path.exists():
            raise SystemExit(f"immutable output already exists: {path}")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    command = [
        sys.executable,
        str(root / "analysis" / "run_fr_b3_catchability.py"),
        "--config",
        str(args.config),
        "--study",
        "factorial",
        "--dry-run",
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    design = json.loads(completed.stdout)
    if design.get("episode_count") != EXPECTED_EPISODES:
        raise SystemExit(
            f"dry-run has {design.get('episode_count')} episodes; expected {EXPECTED_EPISODES}"
        )
    if design.get("development_limits") != {"max_cells": None, "max_seeds": None}:
        raise SystemExit("dry-run unexpectedly contains development limits")
    if (design.get("condition_count"), design.get("seed_count")) != (27, 64):
        raise SystemExit("dry-run does not contain 27 cells and 64 seeds")
    print(
        json.dumps(
            {
                "preflight_passed": True,
                "branch": branch,
                "commit": commit,
                "episodes": EXPECTED_EPISODES,
                "python": sys.version.split()[0],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
