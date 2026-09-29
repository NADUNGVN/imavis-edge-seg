"""Generate the deployment-matched router figure from canonical replay JSON.

The plot reports macro quality and observed mean route cost at each measured
budget.  Each plotted point is first averaged across the five evaluation
splits within a training run, then averaged across Runs A--C.  Whiskers show
the full range across the three run-level macro values; they are descriptive
and are not confidence intervals.

Inputs:
  reports/router_deployment_matched_20260929/run_{a,b,c}_{e1,e3}_replay.json
  reports/router_deployment_matched_20260929/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

BACKENDS = ("E3", "E1")
BACKEND_LABELS = {
    "E3": "(a) E3 · TensorRT/CUDA",
    "E1": "(b) E1 · Hailo-8",
}
RUNS = ("a", "b", "c")
POLICIES = {
    "calibrated_risk": {
        "label": "A · rank escalation",
        "color": "#6B7280",
        "marker": "o",
        "linestyle": ":",
    },
    "risk_latency_constrained": {
        "label": "D · condition-specific",
        "color": "#0072B2",
        "marker": "s",
        "linestyle": "-",
    },
    "pooled_risk_latency_constrained": {
        "label": "D · pooled calibration",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "--",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1]
        / "generated"
        / "fig5_router_deployment_matched",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def compare(delta: float, tolerance: float = 1e-12) -> str:
    if delta > tolerance:
        return "win"
    if delta < -tolerance:
        return "loss"
    return "tie"


def load_replays(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    evidence_dir = repo_root / "reports" / "router_deployment_matched_20260929"
    by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for backend in BACKENDS:
        for run in RUNS:
            path = evidence_dir / f"run_{run}_{backend.lower()}_replay.json"
            payload = read_json(path)
            metadata = payload["_metadata"]
            if metadata["device_id"] != backend:
                raise RuntimeError(f"Backend mismatch in {path}")
            if metadata["risk_feature"] != (
                "mean softmax entropy over all output pixels (deployment-matched v1)"
            ):
                raise RuntimeError(f"Non-canonical risk feature in {path}")
            by_backend[backend].append(payload)
    return by_backend


def run_macro_points(
    payload: dict[str, Any], policy: str
) -> list[dict[str, float]]:
    budgets = payload["_metadata"]["budget_grid_ms"]
    splits = [name for name in payload if name != "_metadata"]
    if len(splits) != 5:
        raise RuntimeError("Expected Cityscapes and four ACDC condition splits")
    result = []
    for budget in budgets:
        points = [
            payload[split]["e2e_aware"][policy][str(budget)] for split in splits
        ]
        result.append(
            {
                "budget_ms": float(budget),
                "mean_latency_ms": float(
                    np.mean([point["mean_latency_ms"] for point in points])
                ),
                "achieved_miou": float(
                    np.mean([point["achieved_miou"] for point in points])
                ),
                "violating_splits": int(
                    sum(point["violation_rate"] > 0.0 for point in points)
                ),
            }
        )
    return result


def aggregate_curve(
    payloads: list[dict[str, Any]], policy: str
) -> dict[str, list[float]]:
    run_points = [run_macro_points(payload, policy) for payload in payloads]
    if len({len(points) for points in run_points}) != 1:
        raise RuntimeError("Run budget grids have different lengths")
    values: dict[str, list[float]] = defaultdict(list)
    for index in range(len(run_points[0])):
        x = np.asarray([points[index]["mean_latency_ms"] for points in run_points])
        y = np.asarray([points[index]["achieved_miou"] for points in run_points])
        values["budget_ms"].append(run_points[0][index]["budget_ms"])
        values["x_mean"].append(float(x.mean()))
        values["x_min"].append(float(x.min()))
        values["x_max"].append(float(x.max()))
        values["y_mean"].append(float(y.mean()))
        values["y_min"].append(float(y.min()))
        values["y_max"].append(float(y.max()))
    return dict(values)


def comparison_summary(
    payloads_by_backend: dict[str, list[dict[str, Any]]], policy: str
) -> dict[str, Any]:
    deltas_fair: list[float] = []
    deltas_all: list[float] = []
    policy_violating = 0
    baseline_violating = 0
    for payloads in payloads_by_backend.values():
        for payload in payloads:
            for split, split_payload in payload.items():
                if split == "_metadata":
                    continue
                by_policy = split_payload["e2e_aware"]
                for budget, baseline in by_policy["calibrated_risk"].items():
                    proposed = by_policy[policy][budget]
                    delta = proposed["achieved_miou"] - baseline["achieved_miou"]
                    deltas_all.append(delta)
                    baseline_violating += baseline["violation_rate"] > 0.0
                    policy_violating += proposed["violation_rate"] > 0.0
                    if baseline["violation_rate"] == 0.0:
                        deltas_fair.append(delta)
    counts = {name: 0 for name in ("win", "tie", "loss")}
    for delta in deltas_fair:
        counts[compare(delta)] += 1
    return {
        "total_cells": len(deltas_all),
        "fair_cells": len(deltas_fair),
        "wins": counts["win"],
        "ties": counts["tie"],
        "losses": counts["loss"],
        "mean_delta_miou_fair": float(np.mean(deltas_fair)),
        "baseline_violating_cells": baseline_violating,
        "policy_violating_cells": policy_violating,
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
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
    payloads_by_backend: dict[str, list[dict[str, Any]]]
) -> tuple[mpl.figure.Figure, dict[str, Any]]:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.35))
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.24, top=0.78, wspace=0.23)
    derived: dict[str, Any] = {"curves": {}, "comparisons": {}}

    for axis, backend in zip(axes, BACKENDS, strict=True):
        derived["curves"][backend] = {}
        for policy, style in POLICIES.items():
            curve = aggregate_curve(payloads_by_backend[backend], policy)
            derived["curves"][backend][policy] = curve
            x = np.asarray(curve["x_mean"])
            y = np.asarray(curve["y_mean"])
            xerr = np.vstack((x - curve["x_min"], np.asarray(curve["x_max"]) - x))
            yerr = np.vstack((y - curve["y_min"], np.asarray(curve["y_max"]) - y))
            axis.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.35,
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.05,
                capsize=2.5,
                elinewidth=0.75,
                alpha=0.98,
                label=style["label"],
            )
        axis.set_title(BACKEND_LABELS[backend], loc="left")
        axis.set_xlabel("Observed mean warm route cost (ms)")
        axis.grid(True, linestyle="--", alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Macro held-out mIoU")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Deployment-matched routing under measured end-to-end cost",
        x=0.5,
        y=0.98,
        ha="center",
        fontsize=10.2,
        fontweight="bold",
    )
    for policy in ("risk_latency_constrained", "pooled_risk_latency_constrained"):
        derived["comparisons"][policy] = comparison_summary(payloads_by_backend, policy)
    derived["aggregation"] = (
        "Each point is the mean of five split-level held-out mIoUs/costs within each "
        "run, then the mean across Runs A--C; whiskers are the run-level min--max range."
    )
    return figure, derived


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    payloads = load_replays(repo_root)
    figure, derived = build_figure(payloads)
    stem = args.output_stem.resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    with stem.with_name(stem.name + "_metrics").with_suffix(".json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(derived, handle, indent=2)
    plt.close(figure)
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
