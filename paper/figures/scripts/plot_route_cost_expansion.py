"""Compare candidate-only p95 latency with measured complete warm-route cost."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

LEVELS = ("tiny", "small", "medium", "large")
BACKENDS = ("E3", "E1")
BACKEND_TITLES = {"E3": "(a) E3 · TensorRT/CUDA", "E1": "(b) E1 · Hailo-8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "figS2_route_cost_expansion",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def candidate_costs(path: Path) -> dict[str, dict[str, float]]:
    result = {backend: {} for backend in BACKENDS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            backend = row["device_id"]
            if backend in result:
                result[backend][row["level"]] = float(row["end_to_end_p95_ms"])
    for backend, values in result.items():
        if set(values) != set(LEVELS):
            raise RuntimeError(f"Missing candidate latency for {backend}: {values}")
    return result


def complete_costs(repo: Path) -> dict[str, dict[str, float]]:
    reports = {
        "E3": read_json(repo / "reports" / "router_overhead_E3_20260922.json"),
        "E1": read_json(repo / "reports" / "router_overhead_E1_20260928.json"),
    }
    result: dict[str, dict[str, float]] = {}
    for backend, report in reports.items():
        entropy_backend = "gpu" if backend == "E3" else "numpy"
        result[backend] = {
            level: float(report["warm"][entropy_backend][f"tiny->{level}"]["median_ms"])
            for level in LEVELS
        }
    return result


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    candidate = candidate_costs(repo / "outputs" / "benchmark_lookup_table.csv")
    complete = complete_costs(repo)

    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.2))
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.78, wspace=0.25)
    x = np.arange(len(LEVELS))
    width = 0.36
    for axis, backend in zip(axes, BACKENDS, strict=True):
        isolated = np.asarray([candidate[backend][level] for level in LEVELS])
        routed = np.asarray([complete[backend][level] for level in LEVELS])
        axis.bar(
            x - width / 2,
            isolated,
            width,
            color="#999999",
            label="Candidate-only p95",
        )
        bars = axis.bar(
            x + width / 2,
            routed,
            width,
            color="#0072B2",
            label="Complete warm route median",
        )
        for bar, ratio in zip(bars, routed / isolated, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(routed) * 0.025,
                f"{ratio:.1f}x",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )
        axis.set_xticks(x, [level.title() for level in LEVELS])
        axis.set_title(BACKEND_TITLES[backend], loc="left", fontweight="bold")
        axis.set_ylabel("Latency (ms)")
        axis.set_ylim(0, max(routed) * 1.20)
        axis.grid(axis="y", color="#DDDDDD", linestyle="--", linewidth=0.55)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Candidate-only latency understates the complete routing path",
        y=0.96,
        fontsize=10.2,
        fontweight="bold",
    )

    stem = args.output_stem.resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)
    metrics = {
        "candidate_only_p95_ms": candidate,
        "complete_warm_route_median_ms": complete,
        "ratio_complete_to_candidate": {
            backend: {
                level: complete[backend][level] / candidate[backend][level]
                for level in LEVELS
            }
            for backend in BACKENDS
        },
        "boundary": "Comparison of two measured totals; not a component decomposition.",
    }
    with stem.with_name(stem.name + "_metrics").with_suffix(".json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
