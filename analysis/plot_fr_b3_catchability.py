#!/usr/bin/env python3
"""Create publication-ready FR-B3 figures from the confirmatory analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from particle_benchmark.io import canonical_json_bytes


INK = "#202124"
GRID = "#DADCE0"
BLUE = "#2463A6"
BLUE_LIGHT = "#D8E8F6"
ORANGE = "#C66A1B"
ORANGE_LIGHT = "#F7DFCB"
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "frb3_gain", [ORANGE, ORANGE_LIGHT, "#FFFFFF", BLUE_LIGHT, BLUE]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(fig: plt.Figure, path_without_suffix: Path) -> list[Path]:
    outputs = [path_without_suffix.with_suffix(".png"), path_without_suffix.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(
        outputs[1],
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return outputs


def _rows_with_predictions(report: dict[str, object]) -> list[dict[str, object]]:
    summaries = {
        str(row["condition_id"]): dict(row)
        for row in report["cell_summaries"]
    }
    comparison = dict(report["predictive_model_comparison"])
    condition_ids = [str(item) for item in comparison["condition_ids"]]
    two = list(comparison["two_axis_loco_predictions"])
    three = list(comparison["three_axis_loco_predictions"])
    rows: list[dict[str, object]] = []
    for index, condition in enumerate(condition_ids):
        row = summaries[condition]
        actual = float(row["mean_paired_gain"])
        row.update(
            {
                "two_axis_loco_prediction": float(two[index]),
                "three_axis_loco_prediction": float(three[index]),
                "two_axis_residual": actual - float(two[index]),
                "three_axis_residual": actual - float(three[index]),
            }
        )
        rows.append(row)
    return rows


def _gain_surface(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    rhos = sorted({float(row["rho"]) for row in rows})
    kappas = sorted({float(row["kappa"]) for row in rows})
    etas = sorted({float(row["eta"]) for row in rows})
    if (len(rhos), len(kappas), len(etas)) != (3, 3, 3):
        raise ValueError("gain surface requires the frozen 3x3x3 design")
    lookup = {
        (float(row["rho"]), float(row["kappa"]), float(row["eta"])): float(
            row["mean_paired_gain"]
        )
        for row in rows
    }
    bound = max(abs(value) for value in lookup.values())
    bound = max(bound, 1e-9)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), sharex=True, sharey=True)
    image = None
    for axis, eta in zip(axes, etas, strict=True):
        matrix = np.asarray(
            [[lookup[(rho, kappa, eta)] for kappa in kappas] for rho in rhos]
        )
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap=GAIN_CMAP,
            vmin=-bound,
            vmax=bound,
            aspect="equal",
        )
        axis.set_title(rf"$\eta={eta:.4g}$", color=INK, fontsize=10)
        axis.set_xticks(range(3), [f"{value:.3g}" for value in kappas])
        axis.set_yticks(range(3), [f"{value:.3g}" for value in rhos])
        axis.set_xlabel(r"Drift/control ratio $\kappa$", color=INK)
        for i in range(3):
            for j in range(3):
                value = matrix[i, j]
                text_color = "white" if abs(value) > 0.58 * bound else INK
                axis.text(j, i, f"{value:+.2f}", ha="center", va="center", color=text_color)
        for spine in axis.spines.values():
            spine.set_color(INK)
    axes[0].set_ylabel(r"Drift SNR $\rho$", color=INK)
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.028, pad=0.035)
    colorbar.set_label("Mean paired coordination gain (captures)", color=INK)
    fig.suptitle("FR-B3 paired coordination-gain surface", color=INK, fontsize=13, y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Shared-summary minus capacity-matched independent; n=64 common seeds per cell",
        ha="center",
        color="#5F6368",
        fontsize=9,
    )
    return _save(fig, output_dir / "paired_gain_surface")


def _prediction_comparison(
    rows: list[dict[str, object]], report: dict[str, object], output_dir: Path
) -> list[Path]:
    actual = np.asarray([float(row["mean_paired_gain"]) for row in rows])
    two = np.asarray([float(row["two_axis_loco_prediction"]) for row in rows])
    three = np.asarray([float(row["three_axis_loco_prediction"]) for row in rows])
    values = np.concatenate((actual, two, three))
    span = max(float(np.ptp(values)), 1.0)
    lower = float(np.min(values) - 0.08 * span)
    upper = float(np.max(values) + 0.08 * span)
    comparison = dict(report["predictive_model_comparison"])
    fig, axis = plt.subplots(figsize=(5.9, 5.2))
    axis.plot([lower, upper], [lower, upper], color=INK, linewidth=1.1, linestyle="--", label="Ideal")
    axis.scatter(
        actual,
        two,
        s=42,
        facecolors="none",
        edgecolors=ORANGE,
        linewidths=1.4,
        marker="s",
        label=f"2-axis LOCO (RMSE {float(comparison['two_axis_loco_rmse']):.2f})",
    )
    axis.scatter(
        actual,
        three,
        s=38,
        color=BLUE,
        edgecolors=INK,
        linewidths=0.35,
        marker="o",
        label=f"3-axis LOCO (RMSE {float(comparison['three_axis_loco_rmse']):.2f})",
    )
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Observed mean paired gain (captures)", color=INK)
    axis.set_ylabel("Held-out predicted gain (captures)", color=INK)
    axis.set_title("Leave-one-cell-out prediction comparison", color=INK, fontsize=12)
    axis.text(
        0.02,
        0.98,
        "27 factorial cells; identical axes and held-out folds",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#5F6368",
    )
    axis.grid(True, color=GRID, linewidth=0.7)
    axis.legend(frameon=False, loc="lower right", fontsize=8.5)
    for spine in axis.spines.values():
        spine.set_color(INK)
    return _save(fig, output_dir / "loco_prediction_comparison")


def create_figures(analysis_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"immutable figure directory already exists: {output_dir}")
    report = json.loads(analysis_path.read_text(encoding="utf-8"))
    if report.get("experiment_id") != "FR-B3-CATCHABILITY":
        raise ValueError("analysis is not FR-B3")
    rows = _rows_with_predictions(report)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: list[Path] = []
    outputs.extend(_gain_surface(rows, output_dir))
    outputs.extend(_prediction_comparison(rows, report, output_dir))

    table_path = output_dir / "cell_summaries_with_predictions.csv"
    fieldnames = list(rows[0])
    with table_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(table_path)

    manifest = {
        "experiment_id": "FR-B3-CATCHABILITY",
        "source_analysis": {
            "path": analysis_path.as_posix(),
            "sha256": _sha256(analysis_path),
        },
        "chart_contracts": [
            {
                "file_stem": "paired_gain_surface",
                "question": "How does paired coordination gain vary across rho, kappa, and eta?",
                "family": "Matrix & Cohort",
                "variant": "three-panel shared-scale heatmap",
                "palette": "hard two-root signed blue-orange with numeric labels",
            },
            {
                "file_stem": "loco_prediction_comparison",
                "question": "Does eta improve held-out prediction over rho and kappa alone?",
                "family": "Uncertainty & Benchmark",
                "variant": "paired prediction scatter against identity",
                "palette": "hard two-root with marker-shape redundancy",
            },
        ],
        "artifacts": {path.name: _sha256(path) for path in outputs},
    }
    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = create_figures(args.analysis, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
