"""Assemble larger V10 qualitative panels from the audited V9 composites."""

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
PANEL_W, PANEL_H = 256, 128
SOURCE_HEADER_H = 34
ROW_LABEL_W = 112
HEADER_H = 38
BLOCK_GAP = 30


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


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


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


def crop_primary(composite: Image.Image, condition: str, panel: str) -> Image.Image:
    row = CONDITIONS.index(condition)
    column = COLUMNS[panel]
    return composite.crop(
        (
            column * PANEL_W,
            SOURCE_HEADER_H + row * PANEL_H,
            (column + 1) * PANEL_W,
            SOURCE_HEADER_H + (row + 1) * PANEL_H,
        )
    )


def crop_hardest(composite: Image.Image, panel: str) -> Image.Image:
    column = COLUMNS[panel]
    return composite.crop(
        (
            column * PANEL_W,
            SOURCE_HEADER_H,
            (column + 1) * PANEL_W,
            SOURCE_HEADER_H + PANEL_H,
        )
    )


def text_center(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, face: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x - (box[2] - box[0]) // 2, y), text, fill="#222222", font=face)


def draw_block(
    canvas: Image.Image,
    top: int,
    headings: list[str],
    columns: list[dict[str, Image.Image]],
    rows: list[tuple[str, str]],
) -> int:
    draw = ImageDraw.Draw(canvas)
    heading_font = font(18, bold=True)
    row_font = font(17, bold=True)
    usable_w = canvas.width - ROW_LABEL_W
    content_w = len(columns) * PANEL_W
    start_x = ROW_LABEL_W + (usable_w - content_w) // 2

    for index, heading in enumerate(headings):
        text_center(
            draw,
            start_x + index * PANEL_W + PANEL_W // 2,
            top + 8,
            heading,
            heading_font,
        )

    for row_index, (label, key) in enumerate(rows):
        y = top + HEADER_H + row_index * PANEL_H
        box = draw.multiline_textbbox((0, 0), label, font=row_font, spacing=2)
        label_h = box[3] - box[1]
        draw.multiline_text(
            (10, y + (PANEL_H - label_h) // 2),
            label,
            fill="#252525",
            font=row_font,
            spacing=2,
        )
        for column_index, panels in enumerate(columns):
            x = start_x + column_index * PANEL_W
            canvas.paste(panels[key], (x, y))
            draw.rectangle(
                (x, y, x + PANEL_W - 1, y + PANEL_H - 1),
                outline="#D4D4D4",
                width=1,
            )
    return top + HEADER_H + len(rows) * PANEL_H


def assemble_split_conditions(
    primary: Image.Image,
    rows: list[tuple[str, str]],
    output: Path,
) -> None:
    groups = (("clean", "fog", "night"), ("rain", "snow"))
    width = ROW_LABEL_W + 3 * PANEL_W
    block_h = HEADER_H + len(rows) * PANEL_H
    height = 2 * block_h + BLOCK_GAP
    canvas = Image.new("RGB", (width, height), "white")
    top = 0
    for group in groups:
        columns = [
            {key: crop_primary(primary, condition, key) for _, key in rows}
            for condition in group
        ]
        top = draw_block(
            canvas,
            top,
            [condition.title() for condition in group],
            columns,
            rows,
        )
        top += BLOCK_GAP
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def assemble_failure_gallery(primary: Image.Image, hardest: Image.Image, output: Path) -> None:
    rows = [
        ("Input", "input"),
        ("Ground\ntruth", "ground_truth"),
        ("Tiny", "tiny"),
        ("Routed\noutput", "selected"),
        ("Error", "error"),
    ]
    width = ROW_LABEL_W + 4 * PANEL_W
    top_block_h = HEADER_H + len(rows) * PANEL_H
    label_h = 28
    detail_gap = 18
    detail_block_h = HEADER_H + 2 * (label_h + PANEL_H) + detail_gap
    canvas = Image.new("RGB", (width, top_block_h + BLOCK_GAP + detail_block_h), "white")

    conditions = ("fog", "night", "rain", "snow")
    columns = [
        {key: crop_primary(primary, condition, key) for _, key in rows}
        for condition in conditions
    ]
    top = draw_block(
        canvas,
        0,
        [f"{condition.title()} - median" for condition in conditions],
        columns,
        rows,
    )

    detail_panels = {key: crop_hardest(hardest, key) for _, key in rows}
    detail_top = top + BLOCK_GAP
    draw = ImageDraw.Draw(canvas)
    heading_font = font(18, bold=True)
    text_center(draw, width // 2, detail_top, "Night - predetermined highest error", heading_font)
    first_label_y = detail_top + HEADER_H
    first_y = first_label_y + label_h
    first_keys = ("input", "ground_truth", "tiny")
    first_labels = ("Input", "Ground truth", "Tiny")
    first_x = ROW_LABEL_W + ((width - ROW_LABEL_W) - 3 * PANEL_W) // 2
    for index, (key, label) in enumerate(zip(first_keys, first_labels, strict=True)):
        x = first_x + index * PANEL_W
        canvas.paste(detail_panels[key], (x, first_y))
        text_center(draw, x + PANEL_W // 2, first_label_y + 3, label, font(15, bold=True))
    second_label_y = first_y + PANEL_H + detail_gap
    second_y = second_label_y + label_h
    second_keys = ("selected", "error")
    second_labels = ("Routed output", "Error overlay")
    second_x = ROW_LABEL_W + ((width - ROW_LABEL_W) - 2 * PANEL_W) // 2
    for index, (key, label) in enumerate(zip(second_keys, second_labels, strict=True)):
        x = second_x + index * PANEL_W
        canvas.paste(detail_panels[key], (x, second_y))
        text_center(draw, x + PANEL_W // 2, second_label_y + 3, label, font(15, bold=True))

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
        if sha256(path) != manifest["outputs"][path.name]:
            raise RuntimeError(f"Source composite hash mismatch: {path}")

    primary = Image.open(primary_path).convert("RGB")
    hardest = Image.open(hardest_path).convert("RGB")
    output_dir = args.output_dir.resolve()
    fig1 = output_dir / "fig1_data_gt_routed_v10.png"
    fig3 = output_dir / "fig3_condition_probe_routes_v10.png"
    fig_s1 = output_dir / "figS1_failure_gallery_v10.png"
    assemble_split_conditions(
        primary,
        [("Input", "input"), ("Ground\ntruth", "ground_truth"), ("Routed\noutput", "selected")],
        fig1,
    )
    assemble_split_conditions(
        primary,
        [("Raw input", "input"), ("Tiny probe\noutput", "tiny"), ("Selected\noutput", "selected")],
        fig3,
    )
    assemble_failure_gallery(primary, hardest, fig_s1)

    audit = {
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_hashes": {path.name: sha256(path) for path in (primary_path, hardest_path)},
        "layout": {
            "fig1": "three conditions above two conditions; input/ground truth/routed output",
            "fig3": "three conditions above two conditions; raw/tiny probe/selected output",
            "figS1": "four median cases plus enlarged predetermined highest-error detail",
        },
        "outputs": {path.name: sha256(path) for path in (fig1, fig3, fig_s1)},
    }
    audit_path = output_dir / "visual_story_v10_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for path in (fig1, fig3, fig_s1, audit_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
