"""Generate the measured-latency versus FLOPs figure from canonical artifacts.

Inputs:
  outputs/flops_by_level.json
  outputs/benchmark_lookup_table.csv

The main panel uses measured mean end-to-end latency.  The right panel compares
observed capacity scaling with the proportional-FLOPs proxy anchored at tiny.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pace_style_v13 import (
    CANDIDATE_COLORS,
    FULL_WIDTH_MM,
    HARDWARE,
    INK,
    apply_style,
    mm_to_inches,
    save_all,
)

LEVELS = ("tiny", "small", "medium", "large")
LEVEL_LABELS = {level: level.capitalize() for level in LEVELS}
DEVICE_MARKERS = {"E1": "o", "E2": "s", "E3": "^", "E5": "D"}
DEVICE_LABELS = {
    "E1": "E1 · Hailo-8",
    "E2": "E2 · TensorRT",
    "E3": "E3 · TensorRT",
    "E5": "E5 · TensorRT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "fig5_flops_latency_v13",
    )
    return parser.parse_args()


def load_data(repo_root: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    flops_path = repo_root / "outputs" / "flops_by_level.json"
    latency_path = repo_root / "outputs" / "benchmark_lookup_table.csv"
    with flops_path.open(encoding="utf-8") as handle:
        raw_flops = json.load(handle)["supernet"]
    gflops = {level: float(raw_flops[level]) / 1e9 for level in LEVELS}

    latency: dict[str, dict[str, float]] = {}
    with latency_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            device = row["device_id"]
            level = row["level"]
            if device in DEVICE_MARKERS and level in LEVELS:
                latency.setdefault(device, {})[level] = float(row["end_to_end_mean_ms"])

    expected = set(DEVICE_MARKERS)
    if set(latency) != expected or any(set(values) != set(LEVELS) for values in latency.values()):
        raise RuntimeError("Latency LUT must contain all four levels for E1, E2, E3, and E5")
    return gflops, latency


def build_figure(
    gflops: dict[str, float], latency: dict[str, dict[str, float]]
) -> plt.Figure:
    apply_style()
    figure, (axis_main, axis_scale) = plt.subplots(
        1,
        2,
        figsize=(mm_to_inches(FULL_WIDTH_MM), mm_to_inches(91)),
        gridspec_kw={"width_ratios": (2.25, 0.82), "wspace": 0.28},
    )
    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.17, top=0.94)
    x = np.array([gflops[level] for level in LEVELS])

    for device, marker in DEVICE_MARKERS.items():
        y = np.array([latency[device][level] for level in LEVELS])
        axis_main.plot(
            x,
            y,
            color=HARDWARE,
            linewidth=1.15,
            marker=marker,
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.15,
            label=DEVICE_LABELS[device],
        )
        for level, x_value, y_value in zip(LEVELS, x, y, strict=True):
            axis_main.scatter(
                x_value,
                y_value,
                s=24,
                marker=marker,
                facecolor=CANDIDATE_COLORS[level],
                edgecolor=INK,
                linewidth=0.45,
                zorder=3,
            )

    axis_main.set_xscale("log")
    axis_main.set_yscale("log")
    axis_main.set_xlabel("Elastic candidate cost (GFLOPs)")
    axis_main.set_ylabel("Measured mean end-to-end latency (ms)")
    axis_main.set_title("(a) Measured latency is hardware specific", loc="left", fontweight="bold")
    axis_main.grid(True, which="major", linestyle="--", alpha=0.8)
    axis_main.set_xticks(x, [f"{value:.2f}" for value in x])
    axis_main.legend(frameon=False, ncol=2, loc="upper left", handletextpad=0.5, columnspacing=1.0)

    device_positions = np.arange(len(DEVICE_MARKERS))
    observed = np.array(
        [latency[device]["large"] / latency[device]["tiny"] for device in DEVICE_MARKERS]
    )
    flops_ratio = gflops["large"] / gflops["tiny"]
    for position, (device, value) in enumerate(zip(DEVICE_MARKERS, observed, strict=True)):
        axis_scale.hlines(position, 0, value, color="#94A3B8", linewidth=1.4)
        axis_scale.scatter(
            value,
            position,
            s=34,
            marker=DEVICE_MARKERS[device],
            facecolor="white",
            edgecolor=HARDWARE,
            linewidth=1.1,
            zorder=3,
        )
        axis_scale.text(
            value + 2.3,
            position,
            f"{value:.1f}\N{MULTIPLICATION SIGN}",
            ha="left",
            va="center",
            fontsize=6.8,
            color=INK,
        )
    axis_scale.axvline(flops_ratio, color="#D55E00", linestyle=(0, (4, 2)), linewidth=1.35)
    axis_scale.text(
        flops_ratio - 2.0,
        -0.58,
        f"FLOPs {flops_ratio:.1f}\N{MULTIPLICATION SIGN}",
        ha="right",
        va="center",
        fontsize=7.0,
        color="#A94400",
        fontweight="bold",
    )
    axis_scale.set_yticks(device_positions, DEVICE_MARKERS.keys())
    axis_scale.invert_yaxis()
    axis_scale.set_xlabel("Large / tiny scaling factor")
    axis_scale.set_title("(b) FLOPs exaggerates scaling", loc="left", fontweight="bold")
    axis_scale.grid(True, axis="x", linestyle="--", alpha=0.8)
    axis_scale.set_xlim(0, flops_ratio * 1.09)
    axis_scale.spines["left"].set_visible(False)
    axis_scale.tick_params(axis="y", length=0)
    return figure


def main() -> None:
    args = parse_args()
    gflops, latency = load_data(args.repo_root.resolve())
    figure = build_figure(gflops, latency)
    stem = args.output_stem.resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_all(figure, stem)
    plt.close(figure)
    print(f"Wrote {stem.with_suffix('.pdf')}, {stem.with_suffix('.svg')}, and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
