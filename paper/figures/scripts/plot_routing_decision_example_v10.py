"""Render an evidence-backed routing decision explainer from existing artifacts.

The script crops unmodified panels from the audited qualitative composite, reads
candidate-specific risks from its render manifest, and reads measured complete
warm-route costs from the canonical E1/E3 overhead reports.  The illustrative
60 ms budget is not a headline evaluation cell; it is used only to expose how an
identical image and risk vector can yield different feasible sets by backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

LEVELS = ("tiny", "small", "medium", "large")
LEVEL_LABELS = tuple(level.title() for level in LEVELS)
LEVEL_COLORS = {
    "tiny": "#E69F00",
    "small": "#56B4E9",
    "medium": "#009E73",
    "large": "#0072B2",
}
CONDITIONS = ("clean", "fog", "night", "rain", "snow")
COLUMNS = {
    "input": 0,
    "ground_truth": 1,
    "tiny": 2,
    "small": 3,
    "medium": 4,
    "large": 5,
    "selected": 6,
    "error": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--condition", choices=CONDITIONS, default="rain")
    parser.add_argument("--budget-ms", type=float, default=60.0)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1]
        / "generated"
        / "fig7_routing_decision_example_v10",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_panel(composite: Image.Image, condition: str, column: str) -> Image.Image:
    panel_w, panel_h, header_h = 256, 128, 34
    row = CONDITIONS.index(condition)
    col = COLUMNS[column]
    return composite.crop(
        (
            col * panel_w,
            header_h + row * panel_h,
            (col + 1) * panel_w,
            header_h + (row + 1) * panel_h,
        )
    )


def route_costs(report: dict[str, Any], backend: str) -> dict[str, float]:
    entropy_backend = "gpu" if backend == "E3" else "numpy"
    warm = report["warm"][entropy_backend]
    return {level: float(warm[f"tiny->{level}"]["median_ms"]) for level in LEVELS}


def select_candidate(
    risks: dict[str, float], costs: dict[str, float], budget: float, target: float
) -> tuple[str, list[str], str]:
    feasible = [level for level in LEVELS if costs[level] <= budget]
    if not feasible:
        cheapest = min(LEVELS, key=lambda level: costs[level])
        return cheapest, [], "cheapest route (budget below all routes)"
    satisfying = [level for level in feasible if risks[level] <= target]
    if satisfying:
        selected = min(satisfying, key=lambda level: costs[level])
        return selected, feasible, "cheapest feasible candidate meeting risk target"
    selected = min(feasible, key=lambda level: (risks[level], costs[level]))
    return selected, feasible, "lowest predicted risk among feasible candidates"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def image_axis(axis: mpl.axes.Axes, image: Image.Image, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title, loc="left", fontweight="bold", pad=3)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#B8B8B8")
        spine.set_linewidth(0.7)


def cost_axis(
    axis: mpl.axes.Axes,
    costs: dict[str, float],
    budget: float,
    selected: str,
    feasible: list[str],
    backend_title: str,
) -> None:
    y = np.arange(len(LEVELS))
    colors = [LEVEL_COLORS[level] if level in feasible else "#D3D3D3" for level in LEVELS]
    bars = axis.barh(y, [costs[level] for level in LEVELS], color=colors, height=0.58)
    for level, bar in zip(LEVELS, bars, strict=True):
        if level not in feasible:
            bar.set_hatch("////")
            bar.set_edgecolor("#777777")
        if level == selected:
            bar.set_edgecolor("#111111")
            bar.set_linewidth(2.0)
        axis.text(
            costs[level] + max(costs.values()) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{costs[level]:.1f}",
            va="center",
            fontsize=8.0,
        )
    axis.axvline(budget, color="#CC3311", linestyle="--", linewidth=1.3)
    axis.set_yticks(y, LEVEL_LABELS)
    axis.invert_yaxis()
    axis.set_xlabel("Measured complete route cost (ms)")
    axis.set_xlim(0, max(max(costs.values()) * 1.24, budget * 1.16))
    axis.set_title(
        f"{backend_title} · B={budget:.0f} ms",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    axis.grid(axis="x", color="#DDDDDD", linestyle="--", linewidth=0.55)
    axis.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    qualitative_dir = repo / "paper" / "figures" / "qualitative" / "generated"
    manifest_path = qualitative_dir / "render_manifest.json"
    composite_path = qualitative_dir / "fig7_qualitative_grid.png"
    manifest = read_json(manifest_path)
    expected_hash = manifest["outputs"]["fig7_qualitative_grid.png"]
    if sha256(composite_path) != expected_hash:
        raise RuntimeError("Qualitative composite hash does not match render manifest")

    entry = next(item for item in manifest["examples"] if item["condition"] == args.condition)
    risks = {level: float(entry["predicted_risk"][level]) for level in LEVELS}
    target = float(entry["risk_target"])
    composite = Image.open(composite_path).convert("RGB")
    panels = {name: crop_panel(composite, args.condition, name) for name in COLUMNS}

    reports = {
        "E3": read_json(repo / "reports" / "router_overhead_E3_20260922.json"),
        "E1": read_json(repo / "reports" / "router_overhead_E1_20260928.json"),
    }
    costs = {backend: route_costs(report, backend) for backend, report in reports.items()}
    decisions = {
        backend: select_candidate(risks, backend_costs, args.budget_ms, target)
        for backend, backend_costs in costs.items()
    }

    configure_style()
    figure = plt.figure(figsize=(7.25, 7.15))
    outer = figure.add_gridspec(
        3,
        2,
        height_ratios=(0.95, 1.0, 1.0),
        left=0.045,
        right=0.99,
        bottom=0.075,
        top=0.92,
        wspace=0.28,
        hspace=0.62,
    )
    sensing = outer[0, 0].subgridspec(1, 2, wspace=0.08)
    ax_input = figure.add_subplot(sensing[0, 0])
    ax_probe = figure.add_subplot(sensing[0, 1])
    image_axis(ax_input, panels["input"], "(a) Input")
    image_axis(ax_probe, panels["tiny"], "Tiny probe output")

    ax_risk = figure.add_subplot(outer[0, 1])
    y = np.arange(len(LEVELS))
    risk_bars = ax_risk.barh(
        y,
        [risks[level] for level in LEVELS],
        color=[LEVEL_COLORS[level] for level in LEVELS],
        height=0.58,
    )
    for level, bar in zip(LEVELS, risk_bars, strict=True):
        ax_risk.text(
            risks[level] + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{risks[level]:.3f}",
            va="center",
            fontsize=8.0,
        )
    ax_risk.axvline(target, color="#CC3311", linestyle="--", linewidth=1.3)
    ax_risk.set_yticks(y, LEVEL_LABELS)
    ax_risk.invert_yaxis()
    ax_risk.set_xlim(0, max(risks.values()) * 1.23)
    ax_risk.set_xlabel("Predicted pixel error")
    ax_risk.set_title(
        rf"(b) Candidate risk · $\tau$={target:.3f}",
        loc="left",
        fontweight="bold",
        pad=3,
    )
    ax_risk.grid(axis="x", color="#DDDDDD", linestyle="--", linewidth=0.55)
    ax_risk.set_axisbelow(True)

    backend_titles = {
        "E3": "(c) E3 · TensorRT/CUDA",
        "E1": "(d) E1 · Hailo-8",
    }
    for row, backend in enumerate(("E3", "E1"), start=1):
        decision_grid = outer[row, :].subgridspec(
            1, 2, width_ratios=(2.75, 1.0), wspace=0.10
        )
        ax_cost = figure.add_subplot(decision_grid[0, 0])
        ax_output = figure.add_subplot(decision_grid[0, 1])
        selected, feasible, _reason = decisions[backend]
        cost_axis(
            ax_cost,
            costs[backend],
            args.budget_ms,
            selected,
            feasible,
            backend_titles[backend],
        )
        image_axis(ax_output, panels[selected], f"Selected: {selected.title()}")

    figure.suptitle(
        "One shared visual-risk estimate, two hardware-feasible decisions",
        x=0.51,
        y=0.98,
        fontsize=12.0,
        fontweight="bold",
    )
    stem = args.output_stem.resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)

    audit = {
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_composite": str(composite_path.relative_to(repo)),
        "source_composite_sha256": expected_hash,
        "image_id": entry["image_id"],
        "condition": entry["condition"],
        "risk_target": target,
        "predicted_risk": risks,
        "illustrative_budget_ms": args.budget_ms,
        "scope_boundary": "derived policy illustration, not a canonical headline evaluation cell",
        "backends": {
            backend: {
                "route_cost_ms": costs[backend],
                "feasible": decisions[backend][1],
                "selected": decisions[backend][0],
                "reason": decisions[backend][2],
            }
            for backend in ("E3", "E1")
        },
    }
    with stem.with_name(stem.name + "_audit").with_suffix(".json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2)
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
