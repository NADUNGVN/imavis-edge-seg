"""Assemble the V9 visual story from audited qualitative composites.

The script verifies the frozen composite hashes, preserves the source-panel
pixels, and adds only labels, borders, and route annotations while arranging
three publication figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

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
PANEL_SIZE = (256, 128)
SOURCE_HEADER_HEIGHT = 34
ROW_LABEL_WIDTH = 132
HEADER_HEIGHT = 46


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "generated",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    roots = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"))
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def crop_primary(composite: Image.Image, condition: str, panel: str) -> Image.Image:
    row = CONDITIONS.index(condition)
    column = COLUMNS[panel]
    width, height = PANEL_SIZE
    return composite.crop(
        (
            column * width,
            SOURCE_HEADER_HEIGHT + row * height,
            (column + 1) * width,
            SOURCE_HEADER_HEIGHT + (row + 1) * height,
        )
    )


def crop_hardest(composite: Image.Image, panel: str) -> Image.Image:
    column = COLUMNS[panel]
    width, height = PANEL_SIZE
    return composite.crop(
        (
            column * width,
            SOURCE_HEADER_HEIGHT,
            (column + 1) * width,
            SOURCE_HEADER_HEIGHT + height,
        )
    )


def assemble(
    columns: list[tuple[str, dict[str, Image.Image]]],
    rows: list[tuple[str, str]],
    output: Path,
) -> None:
    panel_w, panel_h = PANEL_SIZE
    width = ROW_LABEL_WIDTH + panel_w * len(columns)
    height = HEADER_HEIGHT + panel_h * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    heading_font = font(18, bold=True)
    row_font = font(18, bold=True)

    for index, (heading, _) in enumerate(columns):
        box = draw.textbbox((0, 0), heading, font=heading_font)
        x = ROW_LABEL_WIDTH + index * panel_w + (panel_w - (box[2] - box[0])) // 2
        draw.text((x, 11), heading, fill=(28, 28, 28), font=heading_font)

    for row_index, (row_label, panel_key) in enumerate(rows):
        top = HEADER_HEIGHT + row_index * panel_h
        box = draw.multiline_textbbox((0, 0), row_label, font=row_font, spacing=3)
        label_height = box[3] - box[1]
        draw.multiline_text(
            (12, top + (panel_h - label_height) // 2),
            row_label,
            fill=(35, 35, 35),
            font=row_font,
            spacing=3,
        )
        for column_index, (_, panels) in enumerate(columns):
            panel = panels[panel_key]
            left = ROW_LABEL_WIDTH + column_index * panel_w
            canvas.paste(panel, (left, top))
            draw.rectangle(
                (left, top, left + panel_w - 1, top + panel_h - 1),
                outline=(215, 215, 215),
                width=1,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    source_dir = repo / "paper" / "figures" / "qualitative" / "generated"
    manifest_path = source_dir / "render_manifest.json"
    primary_path = source_dir / "fig7_qualitative_grid.png"
    hardest_path = source_dir / "figS1_hardest_failure.png"
    manifest = read_json(manifest_path)

    for path in (primary_path, hardest_path):
        expected = manifest["outputs"][path.name]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Source composite hash mismatch: {path}")

    primary = Image.open(primary_path).convert("RGB")
    hardest = Image.open(hardest_path).convert("RGB")
    metadata = {entry["condition"]: entry for entry in manifest["examples"] if entry["selection_kind"] == "median_large_error"}
    if any(metadata[name]["selected_candidate"] != "medium" for name in CONDITIONS):
        raise RuntimeError("Expected the frozen displayed operating point to select medium")

    primary_columns: list[tuple[str, dict[str, Image.Image]]] = []
    for condition in CONDITIONS:
        panels = {
            key: crop_primary(primary, condition, key)
            for key in ("input", "ground_truth", "tiny", "selected", "error")
        }
        primary_columns.append((condition.title(), panels))

    output_dir = args.output_dir.resolve()
    fig1 = output_dir / "fig1_data_gt_routed_v9.png"
    fig3 = output_dir / "fig3_condition_probe_routes_v9.png"
    fig_s1 = output_dir / "figS1_failure_gallery_v9.png"
    assemble(
        primary_columns,
        [("Input", "input"), ("Ground\ntruth", "ground_truth"), ("Routed\noutput", "selected")],
        fig1,
    )
    assemble(
        primary_columns,
        [("Raw input", "input"), ("Tiny probe\noutput", "tiny"), ("Selected\noutput", "selected")],
        fig3,
    )

    gallery_conditions = ("fog", "night", "rain", "snow")
    gallery_columns = [
        (f"{condition.title()} - median", dict(primary_columns[CONDITIONS.index(condition)][1]))
        for condition in gallery_conditions
    ]
    gallery_columns.append(
        (
            "Night - highest error",
            {
                key: crop_hardest(hardest, key)
                for key in ("input", "ground_truth", "tiny", "selected", "error")
            },
        )
    )
    assemble(
        gallery_columns,
        [
            ("Input", "input"),
            ("Ground\ntruth", "ground_truth"),
            ("Tiny", "tiny"),
            ("Routed\noutput", "selected"),
            ("Error", "error"),
        ],
        fig_s1,
    )

    audit = {
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_hashes": {
            primary_path.name: sha256(primary_path),
            hardest_path.name: sha256(hardest_path),
        },
        "primary_image_ids": [metadata[name]["image_id"] for name in CONDITIONS],
        "failure_gallery": {
            "median_conditions": list(gallery_conditions),
            "highest_error_image_id": next(
                entry["image_id"]
                for entry in manifest["examples"]
                if entry["selection_kind"] == "hardest_acdc_large_error"
            ),
        },
        "displayed_operating_point": {
            "selected_candidate": "medium",
            "budget_ms": metadata["clean"]["budget_ms"],
            "backend": "E3 TensorRT/CUDA warm complete route",
        },
        "outputs": {path.name: sha256(path) for path in (fig1, fig3, fig_s1)},
    }
    audit_path = output_dir / "visual_story_v9_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for path in (fig1, fig3, fig_s1, audit_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
