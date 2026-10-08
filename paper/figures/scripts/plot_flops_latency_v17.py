"""Single-panel measured-latency versus FLOPs figure (V17).

Inputs:
  outputs/flops_by_level.json
  outputs/benchmark_lookup_table.csv

One panel carries the whole RQ1 message: measured latency per device (solid),
the latency that a FLOPs-proportional proxy anchored at tiny would predict
(dashed, same colour), and an illustrative 10 ms budget with the largest
feasible candidate highlighted on each device.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pace_style_v13 import INK, MUTED, SINGLE_WIDTH_MM, apply_style, mm_to_inches, save_all
from plot_flops_latency_v13 import LEVELS, load_data

DEVICES = ("E1", "E2", "E3", "E5")
DEVICE_COLORS = {"E1": "#D55E00", "E2": "#8E5BB5", "E3": "#1F6FB2", "E5": "#2E8B57"}
DEVICE_MARKERS = {"E1": "o", "E2": "s", "E3": "^", "E5": "D"}
DEVICE_LABELS = {
    "E1": "Hailo-8",
    "E2": "Xavier NX",
    "E3": "AGX Xavier",
    "E5": "Orin Nano",
}
BUDGET_MS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "fig5_flops_latency_v17",
    )
    return parser.parse_args()


def build_figure(gflops: dict[str, float], latency: dict[str, dict[str, float]]) -> plt.Figure:
    apply_style()
    figure, axis = plt.subplots(figsize=(mm_to_inches(SINGLE_WIDTH_MM), mm_to_inches(78)))
    figure.subplots_adjust(left=0.15, right=0.80, bottom=0.15, top=0.97)
    x = np.array([gflops[level] for level in LEVELS])
    flops_ratio = x[-1] / x[0]

    axis.axhline(BUDGET_MS, color=INK, linewidth=0.8, linestyle=(0, (1.5, 1.5)), zorder=1)
    axis.text(x[0] * 0.92, BUDGET_MS * 1.08, "10 ms budget", fontsize=6.6, color=INK, ha="left", va="bottom")

    ends = sorted(DEVICES, key=lambda d: latency[d]["large"])
    label_y: dict[str, float] = {}
    previous = None
    for device in ends:
        value = latency[device]["large"]
        if previous is not None and value < previous * 1.6:
            value = previous * 1.6
        label_y[device] = value
        previous = value

    for device in DEVICES:
        color = DEVICE_COLORS[device]
        y = np.array([latency[device][level] for level in LEVELS])
        proxy = y[0] * x / x[0]
        axis.plot(x, proxy, color=color, linewidth=0.9, linestyle=(0, (4, 2.5)), alpha=0.55, zorder=2)
        axis.plot(
            x,
            y,
            color=color,
            linewidth=1.5,
            marker=DEVICE_MARKERS[device],
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.1,
            zorder=3,
        )
        feasible = [index for index, value in enumerate(y) if value <= BUDGET_MS]
        if feasible:
            pick = feasible[-1]
            axis.scatter(
                x[pick], y[pick], s=34, marker=DEVICE_MARKERS[device],
                facecolor=color, edgecolor=color, zorder=4,
            )
        axis.text(
            x[-1] * 1.18,
            label_y[device],
            f"{DEVICE_LABELS[device]}\n{y[-1] / y[0]:.1f}\N{MULTIPLICATION SIGN} measured",
            fontsize=6.2,
            color=color,
            ha="left",
            va="center",
            linespacing=1.1,
        )

    axis.text(
        x[-1] * 1.18,
        latency["E2"]["tiny"] * flops_ratio * 1.25,
        f"{flops_ratio:.1f}\N{MULTIPLICATION SIGN} if latency scaled\nwith FLOPs (dashed)",
        fontsize=6.2,
        color=MUTED,
        ha="left",
        va="center",
        linespacing=1.1,
    )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(x[0] * 0.8, x[-1] * 1.15)
    axis.set_xticks(x, [f"{level.title()}\n{value:.2f}" for level, value in zip(LEVELS, x, strict=True)])
    axis.minorticks_off()
    axis.set_xlabel("Candidate compute (GFLOPs)")
    axis.set_ylabel("Candidate-only mean latency (ms)")
    axis.grid(True, which="major", linestyle="--", alpha=0.7)
    return figure


def main() -> None:
    args = parse_args()
    gflops, latency = load_data(args.repo_root.resolve())
    figure = build_figure(gflops, latency)
    stem = args.output_stem.resolve()
    save_all(figure, stem)
    plt.close(figure)
    print(f"Wrote {stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
