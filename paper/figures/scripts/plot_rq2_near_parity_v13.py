"""Render the three-run matched-capacity near-parity forest plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pace_style_v13 import (
    FULL_WIDTH_MM,
    INK,
    MUTED,
    PANEL_EDGE,
    apply_style,
    mm_to_inches,
    save_all,
)

CONDITIONS = ("cityscapes", "acdc/fog", "acdc/night", "acdc/rain", "acdc/snow")
LABELS = ("Cityscapes", "Fog", "Night", "Rain", "Snow")
SUPERNET_FILES = (
    "reports/eval_pace_seg_v1_aug_seed0_step100000.json",
    "reports/eval_pace_seg_v1_seed2_step100000.json",
    "reports/eval_pace_seg_v1_aug_seed3_step100000.json",
)
BASELINE_FILES = (
    "reports/eval_baseline_fast_scnn_aug_step100000.json",
    "reports/eval_baseline_fast_scnn_seed1_aug_step100000.json",
    "reports/eval_baseline_fast_scnn_seed2_aug_step100000.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "fig6_rq2_near_parity_v13",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_runs(repo_root: Path) -> tuple[np.ndarray, np.ndarray]:
    supernet_runs = []
    for relative in SUPERNET_FILES:
        payload = read_json(repo_root / relative)
        supernet_runs.append([float(payload["large"][condition]) for condition in CONDITIONS])
    baseline_runs = []
    for relative in BASELINE_FILES:
        payload = read_json(repo_root / relative)
        baseline_runs.append([float(payload[condition]["miou"]) for condition in CONDITIONS])
    return np.asarray(supernet_runs), np.asarray(baseline_runs)


def build_figure(supernet: np.ndarray, baseline: np.ndarray) -> plt.Figure:
    apply_style()
    mean_delta = supernet.mean(axis=0) - baseline.mean(axis=0)
    lower = supernet.min(axis=0) - baseline.max(axis=0)
    upper = supernet.max(axis=0) - baseline.min(axis=0)
    xerr = np.vstack((mean_delta - lower, upper - mean_delta)) * 100.0
    mean_points = mean_delta * 100.0

    figure, axis = plt.subplots(figsize=(mm_to_inches(FULL_WIDTH_MM), mm_to_inches(78)))
    figure.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.91)
    y = np.arange(len(CONDITIONS))
    axis.axvspan(-1.5, 1.5, color="#E7F3EE", zorder=0)
    axis.axvline(-1.5, color="#8DB8A5", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    axis.axvline(1.5, color="#8DB8A5", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    axis.axvline(0.0, color=INK, linewidth=1.25, zorder=1)
    axis.errorbar(
        mean_points,
        y,
        xerr=xerr,
        fmt="o",
        color="#0072B2",
        ecolor="#64748B",
        elinewidth=1.35,
        capsize=4.0,
        markersize=6.4,
        markeredgecolor=INK,
        markeredgewidth=0.6,
        zorder=3,
    )
    for x_value, y_value in zip(mean_points, y, strict=True):
        axis.text(
            x_value + (0.14 if x_value >= 0 else -0.14),
            y_value - 0.16,
            f"{x_value:+.2f}",
            ha="left" if x_value >= 0 else "right",
            va="center",
            fontsize=7.3,
            color="#004C70",
            fontweight="bold",
        )
    axis.set_yticks(y, LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("Large shared-supernet candidate - independent Fast-SCNN (mIoU points)")
    axis.grid(True, axis="x", linestyle="--", alpha=0.85)
    axis.set_axisbelow(True)
    axis.set_xlim(
        min(-4.5, float(lower.min() * 100 - 0.5)),
        max(4.5, float(upper.max() * 100 + 0.5)),
    )
    axis.text(
        0.5,
        1.035,
        "prespecified practical near-parity band: -1.5 to +1.5",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#356A54",
    )
    axis.spines["left"].set_color(PANEL_EDGE)
    figure.text(
        0.985,
        0.025,
        "whiskers: full three-run envelope; descriptive, not a confidence interval",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=MUTED,
    )
    return figure


def main() -> None:
    args = parse_args()
    supernet, baseline = load_runs(args.repo_root.resolve())
    figure = build_figure(supernet, baseline)
    stem = args.output_stem.resolve()
    save_all(figure, stem)
    plt.close(figure)
    print(f"Wrote {stem.with_suffix('.pdf')}, {stem.with_suffix('.svg')}, and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
