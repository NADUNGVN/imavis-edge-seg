"""Render the V15 layout-refined audited image-to-routing pipeline.

The visual reuses unmodified panels from the deterministic qualitative artifact,
reads the probe entropy and candidate calibrators from the canonical Run A dump,
and reads complete E3 warm-route costs from the measured overhead report.
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
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "fig4_complete_routing_decision_v15",
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


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 9.8,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def image_axis(axis: mpl.axes.Axes, image: Image.Image, title: str) -> None:
    axis.imshow(image)
    if title:
        axis.set_title(title, loc="left", fontweight="bold", pad=3)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#A8A8A8")
        spine.set_linewidth(0.65)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    qualitative = repo / "paper" / "figures" / "qualitative" / "generated"
    manifest_path = qualitative / "render_manifest.json"
    composite_path = qualitative / "fig7_qualitative_grid.png"
    dump_path = repo / "reports" / "router_deployment_matched_20260929" / "run_a_per_image.json"
    overhead_path = repo / "reports" / "router_overhead_E3_20260922.json"

    manifest = read_json(manifest_path)
    expected_hash = manifest["outputs"]["fig7_qualitative_grid.png"]
    if sha256(composite_path) != expected_hash:
        raise RuntimeError("Qualitative composite hash does not match render manifest")
    entry = next(item for item in manifest["examples"] if item["condition"] == args.condition)
    if entry["selected_candidate"] != "medium":
        raise RuntimeError("Expected the frozen qualitative operating point to select medium")

    dump = read_json(dump_path)
    split_dump = dump[entry["split"]]
    raw_score = float(split_dump["test_raw_scores"][entry["heldout_position"]])
    risks = {level: float(entry["predicted_risk"][level]) for level in LEVELS}
    risk_target = float(entry["risk_target"])

    overhead = read_json(overhead_path)
    costs = {
        level: float(overhead["warm"]["gpu"][f"tiny->{level}"]["median_ms"])
        for level in LEVELS
    }
    budget = float(entry["budget_ms"])
    feasible = [level for level in LEVELS if costs[level] <= budget + 1e-9]
    if feasible != ["tiny", "small", "medium"]:
        raise RuntimeError(f"Unexpected feasible set: {feasible}")

    composite = Image.open(composite_path).convert("RGB")
    panels = {name: crop_panel(composite, args.condition, name) for name in COLUMNS}

    configure_style()
    figure = plt.figure(figsize=(7.25, 4.78))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 1.0),
        height_ratios=(0.80, 1.20),
        left=0.055,
        right=0.975,
        bottom=0.105,
        top=0.875,
        wspace=0.36,
        hspace=0.30,
    )

    ax_input = figure.add_subplot(grid[0, 0])
    image_axis(ax_input, panels["input"], "")
    ax_input.text(
        0.02,
        -0.16,
        "ACDC/rain · held-out",
        transform=ax_input.transAxes,
        fontsize=6.8,
        color="#444444",
    )

    ax_pre = figure.add_subplot(grid[0, 1])
    ax_pre.axis("off")
    ax_pre.add_patch(
        plt.Rectangle(
            (0.10, 0.10),
            0.80,
            0.74,
            transform=ax_pre.transAxes,
            facecolor="#F2F2F2",
            edgecolor="#777777",
            linewidth=0.9,
        )
    )
    ax_pre.text(
        0.5,
        0.50,
        "resize\n384 x 192\n+ normalize",
        ha="center",
        va="center",
        linespacing=1.35,
    )
    ax_pre.text(0.5, 0.20, "fixed-shape tensor", ha="center", va="center", fontsize=6.6)

    probe_grid = grid[0, 2].subgridspec(2, 1, height_ratios=(1.0, 0.34), hspace=0.12)
    ax_probe = figure.add_subplot(probe_grid[0, 0])
    image_axis(ax_probe, panels["tiny"], "")
    ax_entropy = figure.add_subplot(probe_grid[1, 0])
    ax_entropy.axis("off")
    ax_entropy.add_patch(
        plt.Rectangle(
            (0.08, 0.08),
            0.84,
            0.82,
            transform=ax_entropy.transAxes,
            facecolor="#FFF4D6",
            edgecolor="#E69F00",
            linewidth=0.9,
        )
    )
    ax_entropy.text(0.5, 0.67, "mean entropy", ha="center", va="center", fontsize=6.7)
    ax_entropy.text(
        0.5,
        0.32,
        rf"$s(x)={raw_score:.3f}$",
        ha="center",
        va="center",
        fontweight="bold",
    )

    ax_risk = figure.add_subplot(grid[1, 0])
    y = np.arange(len(LEVELS))
    bars = ax_risk.barh(
        y,
        [risks[level] for level in LEVELS],
        color=[LEVEL_COLORS[level] for level in LEVELS],
        height=0.56,
    )
    for level, bar in zip(LEVELS, bars, strict=True):
        ax_risk.text(
            risks[level] + 0.0015,
            bar.get_y() + bar.get_height() / 2,
            f"{risks[level]:.3f}",
            va="center",
            fontsize=6.5,
        )
    ax_risk.axvline(risk_target, color="#CC3311", linestyle="--", linewidth=1.1)
    ax_risk.set_yticks(y, [level.title() for level in LEVELS])
    ax_risk.invert_yaxis()
    ax_risk.set_xlim(0, max(risks.values()) * 1.22)
    ax_risk.set_xlabel("Predicted error")
    ax_risk.text(
        risk_target,
        0.98,
        rf"$\tau={risk_target:.3f}$",
        transform=ax_risk.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.8,
        color="#9C2F13",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )
    ax_risk.grid(axis="x", color="#DDDDDD", linestyle="--", linewidth=0.5)
    ax_risk.set_axisbelow(True)

    ax_cost = figure.add_subplot(grid[1, 1])
    cost_bars = ax_cost.barh(
        y,
        [costs[level] for level in LEVELS],
        color=[LEVEL_COLORS[level] if level in feasible else "#D3D3D3" for level in LEVELS],
        height=0.56,
    )
    for level, bar in zip(LEVELS, cost_bars, strict=True):
        if level not in feasible:
            bar.set_hatch("////")
            bar.set_edgecolor("#777777")
        if level == "medium":
            bar.set_edgecolor("#111111")
            bar.set_linewidth(1.6)
        ax_cost.text(
            costs[level] + 0.45,
            bar.get_y() + bar.get_height() / 2,
            f"{costs[level]:.1f}",
            va="center",
            fontsize=6.5,
        )
    ax_cost.axvline(budget, color="#CC3311", linestyle="--", linewidth=1.1)
    ax_cost.set_yticks(y, [level.title() for level in LEVELS])
    ax_cost.invert_yaxis()
    ax_cost.set_xlim(0, max(costs.values()) * 1.24)
    ax_cost.set_xlabel("Complete route (ms)")
    ax_cost.text(
        budget,
        0.98,
        rf"$B={budget:.2f}$ ms",
        transform=ax_cost.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6.8,
        color="#9C2F13",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )
    ax_cost.grid(axis="x", color="#DDDDDD", linestyle="--", linewidth=0.5)
    ax_cost.set_axisbelow(True)

    output_grid = grid[1, 2].subgridspec(2, 1, hspace=0.30)
    ax_output = figure.add_subplot(output_grid[0, 0])
    image_axis(ax_output, panels["medium"], "")
    ax_error = figure.add_subplot(output_grid[1, 0])
    image_axis(ax_error, panels["error"], "Output error overlay")

    top_titles = (
        "(a) Raw adverse-condition image",
        "(b) Prepare tiny input",
        "(c) Tiny probe",
    )
    bottom_titles = (
        "(d) Candidate risks",
        "(e) E3 feasibility",
        "(f) Selected: Medium",
    )
    for column, title in enumerate(top_titles):
        position = grid[0, column].get_position(figure)
        figure.text(position.x0, position.y1 + 0.010, title, ha="left", va="bottom", fontsize=8.8, fontweight="bold")
    for column, title in enumerate(bottom_titles):
        position = grid[1, column].get_position(figure)
        figure.text(position.x0, position.y1 + 0.010, title, ha="left", va="bottom", fontsize=8.8, fontweight="bold")

    for x in (0.365, 0.675):
        figure.text(x, 0.690, "→", ha="center", va="center", fontsize=14, color="#555555")
    for x in (0.365, 0.675):
        figure.text(x, 0.505, "→", ha="center", va="center", fontsize=14, color="#555555")
    figure.suptitle(
        "From a raw adverse-condition image to one budget-feasible static output",
        y=0.970,
        fontsize=10.2,
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
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)

    audit = {
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_composite": str(composite_path.relative_to(repo)),
        "source_composite_sha256": expected_hash,
        "source_per_image_dump": str(dump_path.relative_to(repo)),
        "source_overhead_report": str(overhead_path.relative_to(repo)),
        "image_id": entry["image_id"],
        "condition": entry["condition"],
        "heldout_position": entry["heldout_position"],
        "probe_resolution": [384, 192],
        "raw_entropy_nats": raw_score,
        "risk_target": risk_target,
        "predicted_risk": risks,
        "budget_ms": budget,
        "route_cost_ms": costs,
        "feasible_candidates": feasible,
        "selected_candidate": "medium",
        "selection_reason": "no feasible candidate meets tau; choose lowest predicted risk among feasible candidates",
        "confusion_matrix_audit": entry["confusion_matrix_audit"],
    }
    with stem.with_name(stem.name + "_audit").with_suffix(".json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2)
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
