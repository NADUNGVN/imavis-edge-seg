"""Generate the RQ2 three-run near-parity delta plot from evaluation JSON.

The two model families use different seed labels, so the figure does not invent
seed-wise pairing.  Each point is the difference of three-run means; each whisker
is the full observed cross-run envelope [min(supernet)-max(baseline),
max(supernet)-min(baseline)].  It is descriptive, not a confidence interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "fig6_rq2_near_parity_v11",
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


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.8,
            "axes.titlesize": 10.4,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_figure(supernet: np.ndarray, baseline: np.ndarray) -> mpl.figure.Figure:
    configure_style()
    mean_delta = supernet.mean(axis=0) - baseline.mean(axis=0)
    lower = supernet.min(axis=0) - baseline.max(axis=0)
    upper = supernet.max(axis=0) - baseline.min(axis=0)
    xerr = np.vstack((mean_delta - lower, upper - mean_delta)) * 100.0
    mean_points = mean_delta * 100.0

    figure, axis = plt.subplots(figsize=(7.25, 3.8))
    figure.subplots_adjust(bottom=0.24)
    y = np.arange(len(CONDITIONS))
    axis.axvspan(-1.5, 1.5, color="#E8F2EE", zorder=0, label="Prespecified near-parity band")
    axis.axvline(0.0, color="#252525", linewidth=0.9, zorder=1)
    axis.errorbar(
        mean_points,
        y,
        xerr=xerr,
        fmt="o",
        color="#0072B2",
        ecolor="#4B5563",
        elinewidth=1.25,
        capsize=3.5,
        markersize=6.2,
        markeredgecolor="#252525",
        markeredgewidth=0.55,
        zorder=3,
    )
    for x_value, y_value in zip(mean_points, y, strict=True):
        axis.text(
            x_value + (0.14 if x_value >= 0 else -0.14),
            y_value - 0.16,
            f"{x_value:+.2f}",
            ha="left" if x_value >= 0 else "right",
            va="center",
            fontsize=7.5,
            color="#003B5C",
        )
    axis.set_yticks(y, LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("Large supernet \N{MINUS SIGN} independent Fast-SCNN (mIoU points)")
    axis.set_title("Three-run mean differences remain within the near-parity band", loc="left")
    axis.grid(True, axis="x", linestyle="--", alpha=0.85)
    axis.set_axisbelow(True)
    axis.set_xlim(
        min(-4.5, float(lower.min() * 100 - 0.5)), max(4.5, float(upper.max() * 100 + 0.5))
    )
    figure.text(
        0.985,
        0.035,
        "Whiskers: observed cross-run envelope (not a confidence interval)",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    return figure


def main() -> None:
    args = parse_args()
    supernet, baseline = load_runs(args.repo_root.resolve())
    figure = build_figure(supernet, baseline)
    stem = args.output_stem.resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
