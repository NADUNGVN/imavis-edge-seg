"""Assemble V13 qualitative figures from frozen audited prediction panels."""

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
CANDIDATE_COLORS = {
    "tiny": "#E69F00",
    "small": "#56B4E9",
    "medium": "#009E73",
    "large": "#0072B2",
}


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
    names = ("arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    roots = (Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"))
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def crop_panel(composite: Image.Image, condition: str, panel: str) -> Image.Image:
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
        (column * PANEL_W, SOURCE_HEADER_H, (column + 1) * PANEL_W, SOURCE_HEADER_H + PANEL_H)
    )


def center_text(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, face: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x - (box[2] - box[0]) // 2, y), text, fill="#1F2937", font=face)


def paste_panel(canvas: Image.Image, panel: Image.Image, x: int, y: int, *, edge: str = "#CBD5E1", width: int = 2) -> None:
    canvas.paste(panel, (x, y))
    ImageDraw.Draw(canvas).rectangle(
        (x, y, x + panel.width - 1, y + panel.height - 1), outline=edge, width=width
    )


def assemble_fig1(primary: Image.Image, manifest: dict[str, Any], output: Path) -> None:
    gap_x, gap_y = 12, 11
    label_w, header_h = 118, 42
    rows = (("Input", "input"), ("Ground truth", "ground_truth"), ("PACE-Seg", "selected"))
    width = label_w + len(CONDITIONS) * PANEL_W + (len(CONDITIONS) - 1) * gap_x
    height = header_h + len(rows) * PANEL_H + (len(rows) - 1) * gap_y
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(20, bold=True)
    row_font = font(17, bold=True)
    entries = {entry["condition"]: entry for entry in manifest["examples"]}

    for column, condition in enumerate(CONDITIONS):
        x = label_w + column * (PANEL_W + gap_x)
        center_text(draw, x + PANEL_W // 2, 8, condition.title(), title_font)
    for row, (label, key) in enumerate(rows):
        y = header_h + row * (PANEL_H + gap_y)
        draw.text((8, y + PANEL_H // 2 - 10), label, fill="#1F2937", font=row_font)
        for column, condition in enumerate(CONDITIONS):
            x = label_w + column * (PANEL_W + gap_x)
            selected = entries[condition]["selected_candidate"]
            edge = CANDIDATE_COLORS[selected] if key == "selected" else "#CBD5E1"
            paste_panel(canvas, crop_panel(primary, condition, key), x, y, edge=edge, width=3 if key == "selected" else 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def assemble_fig9(primary: Image.Image, output: Path) -> None:
    gap_x, gap_y = 10, 12
    label_w, header_h = 116, 38
    top_keys = ("input", "ground_truth", "tiny", "medium", "large")
    top_labels = ("Input", "Ground truth", "Tiny", "Medium", "Large")
    bottom_keys = ("selected", "error")
    bottom_labels = ("PACE-Seg selected", "Error overlay")
    width = label_w + len(top_keys) * PANEL_W + (len(top_keys) - 1) * gap_x
    block_h = header_h + len(CONDITIONS) * PANEL_H + (len(CONDITIONS) - 1) * gap_y
    canvas = Image.new("RGB", (width, 2 * block_h + 30), "white")
    draw = ImageDraw.Draw(canvas)

    for column, label in enumerate(top_labels):
        x = label_w + column * (PANEL_W + gap_x)
        center_text(draw, x + PANEL_W // 2, 6, label, font(16, bold=True))
    for row, condition in enumerate(CONDITIONS):
        y = header_h + row * (PANEL_H + gap_y)
        draw.text((8, y + PANEL_H // 2 - 10), condition.title(), fill="#1F2937", font=font(17, bold=True))
        for column, key in enumerate(top_keys):
            x = label_w + column * (PANEL_W + gap_x)
            paste_panel(canvas, crop_panel(primary, condition, key), x, y)

    bottom_top = block_h + 30
    content_w = len(bottom_keys) * PANEL_W + gap_x
    content_x = label_w + ((width - label_w) - content_w) // 2
    for column, label in enumerate(bottom_labels):
        x = content_x + column * (PANEL_W + gap_x)
        center_text(draw, x + PANEL_W // 2, bottom_top + 6, label, font(16, bold=True))
    for row, condition in enumerate(CONDITIONS):
        y = bottom_top + header_h + row * (PANEL_H + gap_y)
        draw.text((8, y + PANEL_H // 2 - 10), condition.title(), fill="#1F2937", font=font(17, bold=True))
        for column, key in enumerate(bottom_keys):
            x = content_x + column * (PANEL_W + gap_x)
            paste_panel(canvas, crop_panel(primary, condition, key), x, y, edge="#E66101" if key == "error" else "#009E73")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def examples(primary: Image.Image, hardest: Image.Image) -> list[tuple[str, dict[str, Image.Image]]]:
    result = [
        (f"{condition.title()} - median", {key: crop_panel(primary, condition, key) for key in COLUMNS})
        for condition in CONDITIONS
    ]
    result.append(("Night - highest error", {key: crop_hardest(hardest, key) for key in COLUMNS}))
    return result


def assemble_s1(primary: Image.Image, hardest: Image.Image, output: Path) -> None:
    cases = examples(primary, hardest)
    keys = ("input", "ground_truth", "tiny", "selected", "large", "error")
    labels = ("Input", "Ground truth", "Tiny", "Selected", "Large", "Error")
    label_w, gap_x, gap_y, header_h = 126, 8, 10, 38
    width = label_w + len(keys) * PANEL_W + (len(keys) - 1) * gap_x
    height = header_h + len(cases) * PANEL_H + (len(cases) - 1) * gap_y
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        x = label_w + column * (PANEL_W + gap_x)
        center_text(draw, x + PANEL_W // 2, 6, label, font(15, bold=True))
    for row, (title, panels) in enumerate(cases):
        y = header_h + row * (PANEL_H + gap_y)
        draw.multiline_text((7, y + 43), title.replace(" - ", "\n"), fill="#1F2937", font=font(14, bold=True), spacing=1)
        for column, key in enumerate(keys):
            x = label_w + column * (PANEL_W + gap_x)
            paste_panel(canvas, panels[key], x, y, edge="#E66101" if key == "error" else "#CBD5E1")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    source_dir = repo / "paper" / "figures" / "qualitative" / "generated"
    primary_path = source_dir / "fig7_qualitative_grid.png"
    hardest_path = source_dir / "figS1_hardest_failure.png"
    manifest_path = source_dir / "render_manifest.json"
    manifest = read_json(manifest_path)
    for path in (primary_path, hardest_path):
        if sha256(path) != manifest["outputs"][path.name]:
            raise RuntimeError(f"Source composite hash mismatch: {path}")
    primary = Image.open(primary_path).convert("RGB")
    hardest = Image.open(hardest_path).convert("RGB")
    output_dir = args.output_dir.resolve()
    outputs = {
        "fig1": output_dir / "fig1_visual_domains_v13.png",
        "fig9": output_dir / "fig9_qualitative_capacity_v13.png",
        "figS1": output_dir / "figS1_failure_gallery_v13.png",
    }
    assemble_fig1(primary, manifest, outputs["fig1"])
    assemble_fig9(primary, outputs["fig9"])
    assemble_s1(primary, hardest, outputs["figS1"])
    audit = {
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_hashes": {path.name: sha256(path) for path in (primary_path, hardest_path)},
        "evidence_boundary": (
            "V13 changes only deterministic layout, borders, labels, and candidate badges; "
            "all image and prediction pixels come from the audited V11 sources."
        ),
        "outputs": {path.name: sha256(path) for path in outputs.values()},
    }
    audit_path = output_dir / "visual_story_v13_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for path in (*outputs.values(), audit_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
