"""Plot a compact research-style summary of the four elastic candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
        default=Path(__file__).parents[1] / "generated" / "fig2_candidate_family_v10",
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
            "font.size": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "pace-seg-v10-candidate-family",
        }
    )


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    csv_path = repo / "paper" / "tables" / "candidate_family.csv"
    rows = load_rows(csv_path)

    metrics = [
        ("Input", "resolution_hxw", None, lambda value: value),
        ("Width", "width_multiplier", 1.0, lambda value: f"{float(value):.2f}"),
        ("Depth", "depth_blocks", 6.0, lambda value: f"{int(value)} blocks"),
        ("Parameters", "trainable_params", None, lambda value: f"{int(value) / 1e6:.3f} M"),
        ("Compute", "gflops", None, lambda value: f"{float(value):.3f} GFLOPs"),
        ("E3 p95", "e3_candidate_p95_ms", None, lambda value: f"{float(value):.2f} ms"),
        ("E1 p95", "e1_candidate_p95_ms", None, lambda value: f"{float(value):.2f} ms"),
    ]
    numeric_max = {
        key: max(float(row[key]) for row in rows)
        for _, key, _, _ in metrics
        if key != "resolution_hxw"
    }

    configure_style()
    figure, axis = plt.subplots(figsize=(7.25, 4.15))
    figure.subplots_adjust(left=0.15, right=0.995, bottom=0.06, top=0.95)
    axis.set_xlim(-0.85, 3.52)
    axis.set_ylim(-0.55, 7.55)
    axis.axis("off")

    for row_index, (label, key, fixed_max, formatter) in enumerate(metrics):
        y = 6.15 - row_index
        if row_index % 2 == 0:
            axis.axhspan(y - 0.42, y + 0.42, color="#F4F5F6", zorder=0)
        axis.text(
            -0.78,
            y,
            label,
            ha="left",
            va="center",
            color="#343434",
            fontweight="bold",
        )
        for column, (level, data) in enumerate(zip(LEVELS, rows, strict=True)):
            display = formatter(data[key])
            if key != "resolution_hxw":
                maximum = fixed_max or numeric_max[key]
                fraction = float(data[key]) / maximum
                axis.add_patch(
                    Rectangle(
                        (column - 0.36, y - 0.22),
                        0.72 * fraction,
                        0.44,
                        facecolor=COLORS[level],
                        alpha=0.20,
                        edgecolor="none",
                        zorder=1,
                    )
                )
            axis.text(
                column,
                y,
                display,
                ha="center",
                va="center",
                color="#202020",
                fontsize=8.6,
                zorder=2,
            )

    for column, level in enumerate(LEVELS):
        axis.text(
            column,
            7.16,
            level.title(),
            ha="center",
            va="center",
            fontsize=12.0,
            fontweight="bold",
            color="#202020",
        )
        axis.plot(
            [column - 0.35, column + 0.35],
            [6.78, 6.78],
            color=COLORS[level],
            linewidth=4.0,
            solid_capstyle="round",
        )

    for x in (0.5, 1.5, 2.5):
        axis.plot([x, x], [-0.32, 6.65], color="#DDDDDD", linewidth=0.7)

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
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(figure)

    audit = {
        "source": str(csv_path.relative_to(repo)),
        "source_sha256": sha256(csv_path),
        "candidate_order": list(LEVELS),
        "metrics": [label for label, _, _, _ in metrics],
        "boundary": "E1/E3 values are candidate-only p95 latency, not complete route cost",
        "outputs": {
            suffix: sha256(stem.with_suffix(suffix)) for suffix in (".pdf", ".svg", ".png")
        },
    }
    audit_path = stem.with_name(stem.name + "_audit").with_suffix(".json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {stem.with_suffix('.pdf')}, {stem.with_suffix('.svg')}, and {audit_path}")


if __name__ == "__main__":
    main()
