"""Render a sparse visual path from one audited image to a routed output."""

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
COLORS = {"tiny": "#E69F00", "small": "#56B4E9", "medium": "#009E73", "large": "#0072B2"}
CONDITIONS = ("clean", "fog", "night", "rain", "snow")
COLUMNS = {"input": 0, "ground_truth": 1, "tiny": 2, "small": 3, "medium": 4, "large": 5, "selected": 6, "error": 7}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--condition", choices=CONDITIONS, default="rain")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "fig3_visual_routing_pipeline_v11",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_panel(composite: Image.Image, condition: str, column: str) -> Image.Image:
    panel_w, panel_h, header_h = 256, 128, 34
    row = CONDITIONS.index(condition)
    col = COLUMNS[column]
    return composite.crop((col * panel_w, header_h + row * panel_h, (col + 1) * panel_w, header_h + (row + 1) * panel_h))


def image_axis(axis: mpl.axes.Axes, panel: Image.Image, title: str) -> None:
    axis.imshow(panel)
    axis.set_title(title, loc="left", fontweight="bold", fontsize=9.5, pad=4)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#B0B0B0")
        spine.set_linewidth(0.7)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    qualitative = repo / "paper" / "figures" / "qualitative" / "generated"
    manifest_path = qualitative / "render_manifest.json"
    composite_path = qualitative / "fig7_qualitative_grid.png"
    dump_path = repo / "reports" / "router_deployment_matched_20260929" / "run_a_per_image.json"
    overhead_path = repo / "reports" / "router_overhead_E3_20260922.json"
    manifest = read_json(manifest_path)
    if sha256(composite_path) != manifest["outputs"][composite_path.name]:
        raise RuntimeError("Qualitative composite hash does not match render manifest")
    entry = next(item for item in manifest["examples"] if item["condition"] == args.condition)
    dump = read_json(dump_path)
    raw_score = float(dump[entry["split"]]["test_raw_scores"][entry["heldout_position"]])
    risks = {level: float(entry["predicted_risk"][level]) for level in LEVELS}
    target = float(entry["risk_target"])
    budget = float(entry["budget_ms"])
    overhead = read_json(overhead_path)
    costs = {level: float(overhead["warm"]["gpu"][f"tiny->{level}"]["median_ms"]) for level in LEVELS}
    feasible = [level for level in LEVELS if costs[level] <= budget + 1e-9]
    composite = Image.open(composite_path).convert("RGB")
    raw = crop_panel(composite, args.condition, "input")
    prepared = raw.resize((384, 192), Image.Resampling.BILINEAR)
    probe = crop_panel(composite, args.condition, "tiny")
    selected = crop_panel(composite, args.condition, "selected")

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure = plt.figure(figsize=(7.25, 5.0))
    grid = figure.add_gridspec(2, 4, left=0.04, right=0.985, bottom=0.08, top=0.91, wspace=0.34, hspace=0.54)

    ax_raw = figure.add_subplot(grid[0, 0])
    image_axis(ax_raw, raw, "(a) Raw input")
    ax_pre = figure.add_subplot(grid[0, 1])
    image_axis(ax_pre, prepared, "(b) Tiny input")
    ax_pre.text(0.5, -0.13, "resize + normalize · 384x192", transform=ax_pre.transAxes, ha="center", fontsize=7.2, color="#555555")
    ax_probe = figure.add_subplot(grid[0, 2])
    image_axis(ax_probe, probe, "(c) Tiny probe")

    ax_entropy = figure.add_subplot(grid[0, 3])
    ax_entropy.axis("off")
    ax_entropy.add_patch(plt.Circle((0.5, 0.53), 0.28, transform=ax_entropy.transAxes, facecolor="#FFF4D6", edgecolor="#E69F00", linewidth=1.8))
    ax_entropy.text(0.5, 0.66, "mean entropy", transform=ax_entropy.transAxes, ha="center", fontweight="bold")
    ax_entropy.text(0.5, 0.47, f"{raw_score:.3f}", transform=ax_entropy.transAxes, ha="center", fontsize=17, fontweight="bold", color="#7A4A00")
    ax_entropy.set_title("(d) Visual-risk signal", loc="left", fontweight="bold", fontsize=9.5)

    y = np.arange(4)
    ax_risk = figure.add_subplot(grid[1, 0:2])
    risk_bars = ax_risk.barh(y, [risks[level] for level in LEVELS], color=[COLORS[level] for level in LEVELS], height=0.56)
    ax_risk.axvline(target, color="#CC3311", linestyle="--", linewidth=1.1)
    for level, bar in zip(LEVELS, risk_bars, strict=True):
        ax_risk.text(risks[level] + 0.002, bar.get_y() + bar.get_height() / 2, f"{risks[level]:.3f}", va="center", fontsize=7.5)
    ax_risk.set_yticks(y, [level.title() for level in LEVELS])
    ax_risk.invert_yaxis()
    ax_risk.set_xlim(0, max(risks.values()) * 1.2)
    ax_risk.set_xlabel("Predicted pixel error")
    ax_risk.set_title("(e) Candidate-specific expected error", loc="left", fontweight="bold", fontsize=9.5)
    ax_risk.grid(axis="x", linestyle="--", color="#D5D5D5", linewidth=0.6)
    ax_risk.spines[["top", "right"]].set_visible(False)

    ax_feasible = figure.add_subplot(grid[1, 2])
    ax_feasible.axis("off")
    ax_feasible.set_title("(f) Budget-feasible set", loc="left", fontweight="bold", fontsize=9.5)
    for index, level in enumerate(LEVELS):
        in_budget = level in feasible
        y_pos = 0.80 - index * 0.20
        ax_feasible.add_patch(plt.Rectangle((0.08, y_pos - 0.055), 0.13, 0.11, transform=ax_feasible.transAxes, facecolor=COLORS[level] if in_budget else "white", edgecolor=COLORS[level] if in_budget else "#888888", hatch=None if in_budget else "////", linewidth=1.0))
        ax_feasible.text(0.29, y_pos, level.title(), transform=ax_feasible.transAxes, va="center", fontsize=7.8, fontweight="bold" if level == entry["selected_candidate"] else "normal")
        ax_feasible.text(0.99, y_pos, f"{costs[level]:.1f} ms", transform=ax_feasible.transAxes, va="center", ha="right", fontsize=6.8)
    ax_feasible.text(0.5, 0.02, f"budget = {budget:.2f} ms", transform=ax_feasible.transAxes, ha="center", color="#CC3311", fontweight="bold")

    ax_selected = figure.add_subplot(grid[1, 3])
    image_axis(ax_selected, selected, "(g) Selected output")
    ax_selected.text(0.5, -0.13, entry["selected_candidate"].title(), transform=ax_selected.transAxes, ha="center", fontsize=8.2, fontweight="bold", color=COLORS[entry["selected_candidate"]])

    for x in (0.255, 0.505, 0.755):
        figure.text(x, 0.69, "→", ha="center", va="center", fontsize=15, color="#666666")
    figure.text(0.50, 0.46, "candidate risks + measured route costs", ha="center", fontsize=8.0, color="#555555")
    figure.suptitle("From one raw image to a budget-feasible static segmentation", y=0.985, fontsize=11.0, fontweight="bold")

    stem = args.output_stem.resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(figure)
    audit = {
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_composite_sha256": manifest["outputs"][composite_path.name],
        "source_per_image_dump": str(dump_path.relative_to(repo)),
        "source_overhead_report": str(overhead_path.relative_to(repo)),
        "image_id": entry["image_id"],
        "condition": entry["condition"],
        "raw_entropy_nats": raw_score,
        "risk_target": target,
        "predicted_risk": risks,
        "budget_ms": budget,
        "route_cost_ms": costs,
        "feasible_candidates": feasible,
        "selected_candidate": entry["selected_candidate"],
    }
    stem.with_name(stem.name + "_audit").with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
