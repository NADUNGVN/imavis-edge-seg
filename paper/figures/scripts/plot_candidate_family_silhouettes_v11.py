"""Render the four elastic capacities as source-backed network silhouettes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

LEVELS = ("tiny", "small", "medium", "large")
COLORS = {
    "tiny": "#E69F00",
    "small": "#56B4E9",
    "medium": "#009E73",
    "large": "#0072B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path(__file__).parents[1] / "generated" / "fig2_candidate_family_v11",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_level = {row["candidate"]: row for row in rows}
    if set(by_level) != set(LEVELS):
        raise RuntimeError(f"Unexpected candidate rows: {sorted(by_level)}")
    return [by_level[level] for level in LEVELS]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "pace-seg-v11-candidate-family",
        }
    )


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    source = repo / "paper" / "tables" / "candidate_family.csv"
    rows = load_rows(source)
    configure_style()

    figure, axes = plt.subplots(1, 4, figsize=(7.25, 3.55), sharey=True)
    figure.subplots_adjust(left=0.025, right=0.99, bottom=0.08, top=0.80, wspace=0.10)
    max_depth = max(int(row["depth_blocks"]) for row in rows)

    for axis, level, row in zip(axes, LEVELS, rows, strict=True):
        color = COLORS[level]
        width = float(row["width_multiplier"])
        depth = int(row["depth_blocks"])
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.set_title(level.title(), fontsize=12, fontweight="bold", color="#202020", pad=8)

        # Increasing feature-stack width and depth form a compact network silhouette.
        block_w = 0.30 + 0.48 * width
        block_h = 0.055
        gap = 0.018
        total_h = depth * block_h + (depth - 1) * gap
        start_y = 0.48 + total_h / 2 - block_h
        for index in range(depth):
            x = 0.5 - block_w / 2 + index * 0.009
            y = start_y - index * (block_h + gap)
            alpha = 0.26 + 0.62 * (index + 1) / max_depth
            axis.add_patch(
                FancyBboxPatch(
                    (x, y),
                    block_w - index * 0.018,
                    block_h,
                    boxstyle="round,pad=0.01,rounding_size=0.018",
                    linewidth=1.0,
                    edgecolor=color,
                    facecolor=mpl.colors.to_rgba(color, alpha),
                )
            )
        axis.annotate(
            "",
            xy=(0.50, 0.40 - max(0, depth - 4) * 0.022),
            xytext=(0.50, 0.30),
            arrowprops={"arrowstyle": "-|>", "color": "#6B7280", "lw": 0.9},
        )

        axis.text(0.5, 0.86, row["resolution_hxw"], ha="center", fontweight="bold")
        axis.text(
            0.5,
            0.79,
            f"width {float(row['width_multiplier']):.2f}  |  depth {depth}",
            ha="center",
            color="#4B5563",
            fontsize=8.2,
        )

        axis.add_patch(
            Rectangle(
                (0.035, 0.035),
                0.93,
                0.21,
                facecolor="#F4F5F5",
                edgecolor="#D2D5D8",
                linewidth=0.8,
            )
        )
        axis.text(
            0.5,
            0.195,
            f"{int(row['trainable_params']) / 1e6:.3f} M params",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=8.1,
        )
        axis.text(
            0.5,
            0.135,
            f"{float(row['gflops']):.3f} GFLOPs",
            ha="center",
            va="center",
            fontsize=7.8,
        )
        axis.text(
            0.5,
            0.075,
            f"E3 {float(row['e3_candidate_p95_ms']):.2f} ms  |  "
            f"E1 {float(row['e1_candidate_p95_ms']):.2f} ms",
            ha="center",
            va="center",
            fontsize=7.2,
            color="#424242",
        )

    figure.suptitle(
        "One shared parameter family, four increasing static capacities",
        fontsize=11.3,
        fontweight="bold",
        y=0.965,
    )
    stem = args.output_stem.resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Date": None},
    )
    figure.savefig(stem.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(figure)

    audit = {
        "source": str(source.relative_to(repo)),
        "source_sha256": sha256(source),
        "candidate_order": list(LEVELS),
        "encoding": {
            "silhouette_width": "width_multiplier",
            "silhouette_blocks": "depth_blocks",
            "labels": ["resolution", "params", "gflops", "E3 p95", "E1 p95"],
        },
        "boundary": "E1/E3 values are candidate-only p95 latency",
        "archify_source": "paper/figures/archify/fig2_candidate_family_v11/candidate.json",
        "outputs": {
            suffix: sha256(stem.with_suffix(suffix)) for suffix in (".pdf", ".svg", ".png")
        },
    }
    audit_path = stem.with_name(stem.name + "_audit").with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {stem.with_suffix('.pdf')}, {stem.with_suffix('.svg')}, and {audit_path}")


if __name__ == "__main__":
    main()
