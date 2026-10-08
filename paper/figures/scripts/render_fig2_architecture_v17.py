"""Render the V15 layout-refined PACE-Seg architecture figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from pace_style_v13 import (
    CANDIDATE_COLORS,
    CANDIDATE_LABELS,
    FULL_WIDTH_MM,
    HARDWARE,
    INK,
    MUTED,
    PANEL_EDGE,
    PANEL_FILL,
    apply_style,
    mm_to_inches,
    save_all,
)

LEVELS = ("tiny", "small", "medium", "large")
REQUIRED_NODE_IDS = {
    "image",
    "stem",
    "stage1",
    "stage2",
    "stage3",
    "reduce3",
    "add2",
    "fuse2",
    "reduce2",
    "add1",
    "fuse1",
    "reduce1",
    "add0",
    "fuse0",
    "classifier",
    "logits",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--candidates", type=Path, default=Path("paper/tables/candidate_family.csv"),
                        help="candidate card table (landscape V20: paper/tables/candidate_family_landscape.csv)")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "fig2_architecture_v15",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ir(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        ir = yaml.safe_load(stream)
    node_ids = {node["id"] for node in ir["nodes"]}
    missing = REQUIRED_NODE_IDS - node_ids
    if missing:
        raise RuntimeError(f"Architecture IR is missing verified nodes: {sorted(missing)}")
    return ir


def load_candidates(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["candidate"]: row for row in csv.DictReader(stream)}
    if tuple(rows) != LEVELS:
        raise RuntimeError(f"Unexpected candidate order in {path}: {tuple(rows)}")
    return rows


def rounded_box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = "white",
    edge: str = PANEL_EDGE,
    linewidth: float = 0.9,
    radius: float = 0.012,
    zorder: float = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 1.0,
    connectionstyle: str = "arc3",
    zorder: float = 3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
            clip_on=False,
            zorder=zorder,
        )
    )


def label_box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    *,
    face: str,
    edge: str,
    title_size: float = 7.6,
    subtitle_size: float = 5.8,
) -> None:
    rounded_box(ax, x, y, w, h, face=face, edge=edge)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        transform=ax.transAxes,
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.25,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=MUTED,
        transform=ax.transAxes,
        zorder=4,
    )


def draw_elastic_controls(ax: Axes) -> None:
    rounded_box(ax, 0.035, 0.855, 0.93, 0.115, face="#F8FAFC", edge=PANEL_EDGE)
    ax.text(
        0.052,
        0.935,
        r"Elastic configuration $\ell$",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        va="center",
    )

    chips = (
        (0.265, r"resolution $R_\ell$", "input spatial scale"),
        (0.495, r"width $w_\ell$", "active channel prefix"),
        (0.725, r"depth $d_\ell$", "active block prefix"),
    )
    for x, title, subtitle in chips:
        rounded_box(ax, x, 0.872, 0.200, 0.078, face="white", edge="#94A3B8")
        ax.text(
            x + 0.100,
            0.936,
            title,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            x + 0.100,
            0.913,
            subtitle,
            ha="center",
            va="center",
            fontsize=5.6,
            color=MUTED,
            transform=ax.transAxes,
        )
        if "width" in title:
            for index, level in enumerate(LEVELS):
                ax.plot(
                    [x + 0.030, x + 0.055 + index * 0.027],
                    [0.884 + index * 0.004, 0.884 + index * 0.004],
                    color=CANDIDATE_COLORS[level],
                    linewidth=1.3,
                    zorder=4,
                    transform=ax.transAxes,
                    solid_capstyle="round",
                )
        if "depth" in title:
            for index in range(6):
                ax.add_patch(
                    Rectangle(
                        (x + 0.037 + index * 0.021, 0.880),
                        0.014,
                        0.012,
                        facecolor=CANDIDATE_COLORS["large"] if index >= 4 else "#CBD5E1",
                        edgecolor="#94A3B8",
                        linewidth=0.3,
                        transform=ax.transAxes,
                        zorder=4,
                    )
                )


def draw_nested_input(ax: Axes, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, face="white", edge=PANEL_EDGE)
    scales = (0.48, 0.64, 0.82, 1.0)
    for level, scale in zip(LEVELS, scales, strict=True):
        rw, rh = w * 0.80 * scale, h * 0.63 * scale * 0.42  # 2:1 landscape (W x H) inputs
        rx = x + (w - rw) / 2
        ry = y + 0.10 * h + (h * 0.67 - rh) / 2
        ax.add_patch(
            Rectangle(
                (rx, ry),
                rw,
                rh,
                fill=False,
                edgecolor=CANDIDATE_COLORS[level],
                linewidth=1.0,
                transform=ax.transAxes,
                zorder=3,
            )
        )
    ax.text(
        x + w / 2,
        y + h * 0.86,
        "RGB input",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        x + w / 2,
        y + h * 0.02,
        r"$R_\ell$",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
        transform=ax.transAxes,
    )


def draw_architecture(ax: Axes) -> None:
    ax.text(
        0.025,
        0.825,
        "(a)",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.066,
        0.825,
        "One shared elastic encoder-decoder",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.035,
        0.528,
        r"elastic axes:  input resolution $R_\ell$  ·  active channel prefix $w_\ell$  ·  active block prefix $d_\ell$",
        transform=ax.transAxes,
        fontsize=6.8,
        color=MUTED,
        va="top",
    )

    y = 0.555
    h = 0.165
    draw_nested_input(ax, 0.035, y, 0.085, h)
    label_box(ax, 0.140, y, 0.070, h, "Stem", "3x3, stride 2", face="#EFF6FF", edge="#60A5FA")
    label_box(ax, 0.232, y, 0.088, h, "Stage 1", r"DW $\times d_\ell$; down 2", face="#ECFEFF", edge="#22A6B3")
    label_box(ax, 0.342, y, 0.088, h, "Stage 2", r"DW $\times d_\ell$; down 2", face="#ECFEFF", edge="#22A6B3")
    label_box(ax, 0.452, y, 0.088, h, "Stage 3", r"DW $\times d_\ell$; down 2", face="#FFF7ED", edge="#D97706")
    label_box(ax, 0.565, y, 0.085, h, "Decode 2", "1x1; up 2; +; DW", face="#F0FDF4", edge="#16A34A", title_size=7.2)
    label_box(ax, 0.672, y, 0.085, h, "Decode 1", "1x1; up 2; +; DW", face="#F0FDF4", edge="#16A34A", title_size=7.2)
    label_box(ax, 0.779, y, 0.085, h, "Decode 0", "1x1; up 2; +; DW", face="#F0FDF4", edge="#16A34A", title_size=7.2)

    trunk = ((0.120, 0.140), (0.210, 0.232), (0.320, 0.342), (0.430, 0.452), (0.540, 0.565), (0.650, 0.672), (0.757, 0.779), (0.864, 0.888))
    for left, right in trunk:
        arrow(ax, (left, y + h / 2), (right, y + h / 2), linewidth=0.9)

    # Verified additive skips in PaceSegSupernet.forward().
    skip_y = y + h + 0.030
    for x1, x2, lift in ((0.175, 0.8215, 0.00), (0.276, 0.7145, 0.018), (0.386, 0.6075, 0.036)):
        ax.plot(
            [x1, x1, x2, x2],
            [y + h, skip_y + lift, skip_y + lift, y + h],
            color="#64748B",
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            transform=ax.transAxes,
            zorder=2,
        )
        arrow(ax, (x2, y + h + 0.002), (x2, y + h - 0.018), color="#64748B", linewidth=0.75)
    ax.text(0.66, 0.800, "additive encoder skips", fontsize=6.2, color=MUTED, transform=ax.transAxes)

    label_box(ax, 0.888, y, 0.070, h, "Head", "up 2; 1x1", face="#F0FDF4", edge="#16A34A", title_size=6.7)

def draw_candidate_card(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    level: str,
    row: dict[str, str],
) -> None:
    color = CANDIDATE_COLORS[level]
    rounded_box(ax, x, y, w, h, face="white", edge=color, linewidth=1.25)
    ax.add_patch(
        Rectangle(
            (x, y + h - 0.024),
            w,
            0.024,
            facecolor=color,
            edgecolor="none",
            transform=ax.transAxes,
            zorder=3,
        )
    )
    ax.text(
        x + 0.012,
        y + h - 0.043,
        CANDIDATE_LABELS[level],
        fontsize=8.0,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        x + w - 0.012,
        y + h - 0.043,
        row["resolution_hxw"],
        fontsize=6.6,
        fontweight="bold",
        transform=ax.transAxes,
        ha="right",
        va="center",
    )

    depth = int(row["depth_blocks"])
    width = float(row["width_multiplier"])
    base_x = x + 0.015
    base_y = y + 0.112
    block_w = 0.009 + 0.018 * width
    for stage in range(3):
        sx = base_x + stage * 0.042
        for block in range(depth):
            by = base_y + block * 0.008
            ax.add_patch(
                Rectangle(
                    (sx, by),
                    block_w,
                    0.0065,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.28 + 0.08 * stage,
                    linewidth=0.35,
                    transform=ax.transAxes,
                    zorder=3,
                )
            )
    ax.text(
        x + 0.015,
        y + 0.087,
        f"$w={width:.2f}$   $d={depth}$",
        fontsize=6.4,
        color=MUTED,
        transform=ax.transAxes,
    )
    params_m = int(row["trainable_params"]) / 1_000_000
    ax.text(
        x + 0.015,
        y + 0.062,
        f"{params_m:.3f} M params",
        fontsize=6.2,
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.015,
        y + 0.037,
        f"{float(row['gflops']):.3f} GFLOPs",
        fontsize=6.2,
        transform=ax.transAxes,
    )
    ax.text(
        x + 0.015,
        y + 0.012,
        f"AGX {float(row['e3_candidate_p95_ms']):.2f} | Hailo {float(row['e1_candidate_p95_ms']):.2f} ms",
        fontsize=6.0,
        color=HARDWARE,
        transform=ax.transAxes,
    )


def draw_static_candidates(ax: Axes, candidates: dict[str, dict[str, str]]) -> None:
    ax.text(
        0.025,
        0.475,
        "(b)",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.066,
        0.475,
        "Four related static candidates from one trained parameter family",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    ax.text(
        0.055,
        0.392,
        "slice active channels\nkeep stage prefixes\nfreeze one resolution",
        ha="left",
        va="center",
        fontsize=6.5,
        color=MUTED,
        linespacing=1.25,
        transform=ax.transAxes,
    )
    arrow(ax, (0.190, 0.392), (0.224, 0.392), color="#64748B", linewidth=1.0)
    ax.text(0.055, 0.243, "last card line:\ncandidate-only\np95 latency", ha="left", va="center",
            fontsize=6.0, color=HARDWARE, linespacing=1.2, transform=ax.transAxes)

    x_positions = (0.230, 0.415, 0.600, 0.785)
    for x, level in zip(x_positions, LEVELS, strict=True):
        draw_candidate_card(ax, x, 0.168, 0.165, 0.217, level, candidates[level])

    rounded_box(ax, 0.230, 0.012, 0.720, 0.122, face=PANEL_FILL, edge=PANEL_EDGE)
    ax.text(
        0.248,
        0.100,
        "Static export and backend compilation",
        fontsize=7.2,
        fontweight="bold",
        transform=ax.transAxes,
        va="center",
    )
    label_box(ax, 0.530, 0.050, 0.105, 0.054, "ONNX", "fixed shape", face="white", edge="#94A3B8", title_size=7.0)
    label_box(ax, 0.705, 0.082, 0.140, 0.038, "TensorRT GPU", "AGX / NX / Orin", face="#EFF6FF", edge="#3B82F6", title_size=6.5, subtitle_size=5.2)
    label_box(ax, 0.705, 0.024, 0.140, 0.038, "Hailo-8", "HEF", face="#FFF7ED", edge="#D97706", title_size=6.5, subtitle_size=5.2)
    arrow(ax, (0.638, 0.083), (0.702, 0.101), color=HARDWARE, linewidth=0.9)
    arrow(ax, (0.638, 0.072), (0.702, 0.043), color=HARDWARE, linewidth=0.9)


def build_figure(candidates: dict[str, dict[str, str]]) -> plt.Figure:
    apply_style()
    plt.rcParams["savefig.bbox"] = None
    figure = plt.figure(figsize=(mm_to_inches(FULL_WIDTH_MM), mm_to_inches(118)))
    axis = figure.add_axes((0.01, 0.01, 0.98, 0.98 / 0.84))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    draw_architecture(axis)
    draw_static_candidates(axis, candidates)
    return figure


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    ir_path = repo / "paper" / "figures" / "v13" / "architecture" / "pace_seg_exact.yaml"
    candidate_path = repo / args.candidates
    source_paths = (
        repo / "src" / "imavis_edge_seg" / "models" / "supernet.py",
        repo / "src" / "imavis_edge_seg" / "models" / "blocks.py",
        repo / "src" / "imavis_edge_seg" / "models" / "subnet.py",
        repo / "src" / "imavis_edge_seg" / "config.py",
    )
    load_ir(ir_path)
    candidates = load_candidates(candidate_path)
    figure = build_figure(candidates)
    output_stem = args.output_stem.resolve()
    save_all(figure, output_stem)
    plt.close(figure)

    audit = {
        "claim": (
            "One shared elastic encoder-decoder varies resolution, active channel prefixes, "
            "and stage-prefix depth, then exports four independent compiler-safe subnet graphs."
        ),
        "architecture_ir": str(ir_path.relative_to(repo)),
        "candidate_table": str(candidate_path.relative_to(repo)),
        "source_hashes": {str(path.relative_to(repo)): sha256(path) for path in source_paths},
        "artifact_hashes": {
            suffix: sha256(output_stem.with_suffix(suffix)) for suffix in (".pdf", ".svg", ".png")
        },
    }
    audit_path = output_stem.with_name(output_stem.name + "_audit").with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")
    print(f"Wrote {output_stem.with_suffix('.svg')}")
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
