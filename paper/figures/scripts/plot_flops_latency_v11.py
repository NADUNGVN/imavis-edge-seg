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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

LEVELS = ("tiny", "small", "medium", "large")
LEVEL_LABELS = {level: level.capitalize() for level in LEVELS}
LEVEL_COLORS = {
    "tiny": "#0072B2",
    "small": "#56B4E9",
    "medium": "#E69F00",
    "large": "#D55E00",
}
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
        default=Path(__file__).parents[1] / "generated" / "fig5_flops_latency_v11",
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


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.4,
            "legend.fontsize": 8.1,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_figure(
    gflops: dict[str, float], latency: dict[str, dict[str, float]]
) -> mpl.figure.Figure:
    configure_style()
    figure, (axis_main, axis_scale) = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.85),
        gridspec_kw={"width_ratios": (1.75, 1.0), "wspace": 0.34},
    )
    x = np.array([gflops[level] for level in LEVELS])

    for device, marker in DEVICE_MARKERS.items():
        y = np.array([latency[device][level] for level in LEVELS])
        axis_main.plot(
            x,
            y,
            color="#4B5563",
            linewidth=1.25,
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
                facecolor=LEVEL_COLORS[level],
                edgecolor="#333333",
                linewidth=0.45,
                zorder=3,
            )

    axis_main.set_xscale("log")
    axis_main.set_yscale("log")
    axis_main.set_xlabel("Elastic candidate cost (GFLOPs)")
    axis_main.set_ylabel("Measured mean end-to-end latency (ms)")
    axis_main.set_title("(a) Measured latency by accelerator", loc="left")
    axis_main.grid(True, which="major", linestyle="--", alpha=0.8)
    axis_main.set_xticks(x, [f"{value:.2f}" for value in x])
    axis_main.legend(frameon=False, ncol=2, loc="upper left")

    device_positions = np.arange(len(DEVICE_MARKERS))
    observed = np.array(
        [latency[device]["large"] / latency[device]["tiny"] for device in DEVICE_MARKERS]
    )
    flops_ratio = gflops["large"] / gflops["tiny"]
    width = 0.34
    axis_scale.bar(
        device_positions - width / 2,
        observed,
        width,
        color="#0072B2",
        edgecolor="#252525",
        linewidth=0.55,
        label="Measured latency",
    )
    axis_scale.bar(
        device_positions + width / 2,
        np.full_like(observed, flops_ratio),
        width,
        color="white",
        edgecolor="#D55E00",
        hatch="////",
        linewidth=0.8,
        label="Proportional FLOPs proxy",
    )
    axis_scale.set_xticks(device_positions, DEVICE_MARKERS.keys())
    axis_scale.set_ylabel("Large / tiny scaling factor")
    axis_scale.set_title("(b) Large/tiny scaling", loc="left")
    axis_scale.grid(True, axis="y", linestyle="--", alpha=0.8)
    axis_scale.legend(frameon=False, loc="upper left")
    axis_scale.set_ylim(0, flops_ratio * 1.15)
    for position, value in zip(device_positions, observed, strict=True):
        axis_scale.text(
            position - width / 2,
            value + 2.0,
            f"{value:.1f}\N{MULTIPLICATION SIGN}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    return figure


def main() -> None:
    args = parse_args()
    gflops, latency = load_data(args.repo_root.resolve())
    figure = build_figure(gflops, latency)
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
