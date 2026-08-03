"""Low-compute tests for FR-B3 HPC validation and figure generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from analysis.verify_fr_b3_factorial import validate_factorial_run


POLICIES = (
    "capacity_matched_independent",
    "shared_summary_v2",
    "stationary",
    "full_state_interception_oracle",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_factorial(tmp_path: Path) -> tuple[Path, Path]:
    config = {
        "experiment_id": "FR-B3-CATCHABILITY",
        "protocol_status": "registered",
        "seeds": [101, 102],
        "policies": list(POLICIES),
        "fixed_environment": {"particle_count": 10, "horizon": 3},
        "factorial_axes": {
            "rho": [0.5, 1.0],
            "kappa": [0.5, 1.0],
            "eta": [0.5, 1.0],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    conditions = []
    rows = []
    for ri, rho in enumerate((0.5, 1.0)):
        for ki, kappa in enumerate((0.5, 1.0)):
            for ei, eta in enumerate((0.5, 1.0)):
                condition = f"r{ri}_k{ki}_e{ei}"
                groups = {"rho": rho, "kappa": kappa, "eta": eta}
                conditions.append(
                    {"condition_id": condition, "dimensionless_groups": groups}
                )
                for seed in config["seeds"]:
                    streams = {"brownian_sha256": f"stream-{condition}-{seed}"}
                    for policy in POLICIES:
                        rows.append(
                            {
                                "study": "factorial",
                                "condition_id": condition,
                                "seed": seed,
                                "policy_id": policy,
                                "unique_team_capture_yield": 3,
                                "executed_steps": 3,
                                "dimensionless_groups": groups,
                                "stream_checksums": streams,
                            }
                        )
    summaries = run_dir / "episode_summaries.jsonl"
    summaries.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    design = {
        "experiment_id": "FR-B3-CATCHABILITY",
        "study": "factorial",
        "condition_count": 8,
        "seed_count": 2,
        "policies": list(POLICIES),
        "episode_count": len(rows),
        "development_limits": {"max_cells": None, "max_seeds": None},
        "conditions": conditions,
    }
    design_path = run_dir / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")
    manifest = {
        "experiment_id": "FR-B3-CATCHABILITY",
        "study": "factorial",
        "protocol_status": "registered",
        "complete_frozen_design": True,
        "repository": {
            "full_name": "PuffBear/stochastic-particle-system",
            "branch": "fr-b3-catchability-benchmark",
            "commit_sha": "1" * 40,
        },
        "source_snapshot": {},
        "artifacts": {
            summaries.name: _sha256(summaries),
            design_path.name: _sha256(design_path),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, config_path


def test_factorial_verifier_passes_exact_cartesian_product(tmp_path: Path) -> None:
    run_dir, config = _fake_factorial(tmp_path)
    report = validate_factorial_run(
        run_dir, config, verify_git_snapshot=False
    )
    assert report["validation_passed"] is True
    assert report["row_count"] == 64
    assert report["paired_stream_group_count"] == 16


def test_factorial_verifier_rejects_missing_episode(tmp_path: Path) -> None:
    run_dir, config = _fake_factorial(tmp_path)
    summaries = run_dir / "episode_summaries.jsonl"
    lines = summaries.read_text(encoding="utf-8").splitlines()
    summaries.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    report = validate_factorial_run(
        run_dir, config, verify_git_snapshot=False
    )
    assert report["validation_passed"] is False
    assert any("missing episode keys" in item for item in report["failures"])
    assert any("artifact hash mismatch" in item for item in report["failures"])


def test_publication_figures_render_from_analysis_contract(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from analysis.plot_fr_b3_catchability import create_figures

    summaries = []
    condition_ids = []
    actual = []
    two = []
    three = []
    for ri, rho in enumerate((0.5, 1.0, 2.0)):
        for ki, kappa in enumerate((0.5, 1.0, 2.0)):
            for ei, eta in enumerate((0.5, 1.0, 2.0)):
                condition = f"r{ri}_k{ki}_e{ei}"
                gain = (ri - 1) - 0.5 * (ki - 1) + 0.75 * (ei - 1)
                condition_ids.append(condition)
                actual.append(gain)
                two.append(gain - 0.5 * (ei - 1))
                three.append(gain + 0.03 * (ri - ki))
                summaries.append(
                    {
                        "condition_id": condition,
                        "rho": rho,
                        "kappa": kappa,
                        "eta": eta,
                        "n": 64,
                        "mean_paired_gain": gain,
                        "sample_standard_deviation": 2.0,
                        "positive_seed_count": 40,
                        "zero_seed_count": 5,
                    }
                )
    report = {
        "experiment_id": "FR-B3-CATCHABILITY",
        "cell_summaries": summaries,
        "predictive_model_comparison": {
            "condition_ids": condition_ids,
            "two_axis_loco_predictions": two,
            "three_axis_loco_predictions": three,
            "two_axis_loco_rmse": 0.5,
            "three_axis_loco_rmse": 0.1,
        },
    }
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "figures"
    manifest = create_figures(analysis, output)
    assert len(manifest["artifacts"]) == 5
    for name, digest in manifest["artifacts"].items():
        path = output / name
        assert path.stat().st_size > 0
        assert _sha256(path) == digest
