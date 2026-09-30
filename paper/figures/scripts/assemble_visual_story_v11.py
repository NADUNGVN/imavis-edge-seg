"""Assemble V11 qualitative figures from the audited Run A composites."""

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
LABEL_W = 112
HEADER_H = 38
GAP = 28


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


def center_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    face: ImageFont.ImageFont,
) -> None:
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x - (box[2] - box[0]) // 2, y), text, fill="#202020", font=face)


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


def paste_bordered(canvas: Image.Image, panel: Image.Image, x: int, y: int) -> None:
    canvas.paste(panel, (x, y))
    ImageDraw.Draw(canvas).rectangle(
        (x, y, x + panel.width - 1, y + panel.height - 1), outline="#D0D0D0", width=1
    )


def assemble_fig1(primary: Image.Image, output: Path) -> None:
    rows = (("Input", "input"), ("Ground truth", "ground_truth"), ("PACE-Seg", "selected"))
    width = LABEL_W + len(CONDITIONS) * PANEL_W
    height = HEADER_H + len(rows) * PANEL_H
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, condition in enumerate(CONDITIONS):
        center_text(
            draw,
            LABEL_W + index * PANEL_W + PANEL_W // 2,
            7,
            condition.title(),
            font(19, bold=True),
        )
    for row_index, (label, key) in enumerate(rows):
        y = HEADER_H + row_index * PANEL_H
        draw.text((10, y + PANEL_H // 2 - 10), label, fill="#202020", font=font(17, bold=True))
        for column, condition in enumerate(CONDITIONS):
            paste_bordered(canvas, crop_primary(primary, condition, key), LABEL_W + column * PANEL_W, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def assemble_fig9(primary: Image.Image, output: Path) -> None:
    top_keys = ("input", "ground_truth", "tiny", "medium", "large")
    top_labels = ("Input", "Ground truth", "Tiny", "Medium", "Large")
    bottom_keys = ("selected", "error")
    bottom_labels = ("PACE-Seg selected", "Error overlay")
    width = LABEL_W + len(top_keys) * PANEL_W
    top_h = HEADER_H + len(CONDITIONS) * PANEL_H
    bottom_h = HEADER_H + len(CONDITIONS) * PANEL_H
    canvas = Image.new("RGB", (width, top_h + GAP + bottom_h), "white")
    draw = ImageDraw.Draw(canvas)

    for column, label in enumerate(top_labels):
        center_text(draw, LABEL_W + column * PANEL_W + PANEL_W // 2, 7, label, font(17, bold=True))
    for row, condition in enumerate(CONDITIONS):
        y = HEADER_H + row * PANEL_H
        draw.text((10, y + PANEL_H // 2 - 10), condition.title(), fill="#202020", font=font(17, bold=True))
        for column, key in enumerate(top_keys):
            paste_bordered(canvas, crop_primary(primary, condition, key), LABEL_W + column * PANEL_W, y)

    bottom_top = top_h + GAP
    content_x = LABEL_W + (len(top_keys) * PANEL_W - len(bottom_keys) * PANEL_W) // 2
    for column, label in enumerate(bottom_labels):
        center_text(draw, content_x + column * PANEL_W + PANEL_W // 2, bottom_top + 7, label, font(17, bold=True))
    for row, condition in enumerate(CONDITIONS):
        y = bottom_top + HEADER_H + row * PANEL_H
        draw.text((10, y + PANEL_H // 2 - 10), condition.title(), fill="#202020", font=font(17, bold=True))
        for column, key in enumerate(bottom_keys):
            paste_bordered(canvas, crop_primary(primary, condition, key), content_x + column * PANEL_W, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def example_panels(
    primary: Image.Image, hardest: Image.Image
) -> list[tuple[str, dict[str, Image.Image]]]:
    examples = [
        (
            f"{condition.title()} - median",
            {key: crop_primary(primary, condition, key) for key in COLUMNS},
        )
        for condition in CONDITIONS
    ]
    examples.append(
        (
            "Night - highest error",
            {key: crop_hardest(hardest, key) for key in COLUMNS},
        )
    )
    return examples


def assemble_s1(primary: Image.Image, hardest: Image.Image, output: Path) -> None:
    examples = example_panels(primary, hardest)
    rows = (("Input", "input"), ("Ground truth", "ground_truth"), ("Tiny", "tiny"), ("PACE-Seg", "selected"), ("Error", "error"))
    columns_per_block = 3
    width = LABEL_W + columns_per_block * PANEL_W
    block_h = HEADER_H + len(rows) * PANEL_H
    canvas = Image.new("RGB", (width, 2 * block_h + GAP), "white")
    draw = ImageDraw.Draw(canvas)
    for block in range(2):
        top = block * (block_h + GAP)
        block_examples = examples[block * columns_per_block : (block + 1) * columns_per_block]
        for column, (title, _) in enumerate(block_examples):
            center_text(draw, LABEL_W + column * PANEL_W + PANEL_W // 2, top + 7, title, font(16, bold=True))
        for row_index, (label, key) in enumerate(rows):
            y = top + HEADER_H + row_index * PANEL_H
            draw.text((10, y + PANEL_H // 2 - 10), label, fill="#202020", font=font(16, bold=True))
            for column, (_, panels) in enumerate(block_examples):
                paste_bordered(canvas, panels[key], LABEL_W + column * PANEL_W, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def assemble_s3(primary: Image.Image, hardest: Image.Image, output: Path) -> None:
    examples = example_panels(primary, hardest)
    keys = ("input", "ground_truth", "small", "large", "selected", "error")
    labels = ("Input", "Ground truth", "Small", "Large", "PACE-Seg", "Error")
    rows_per_block = 3
    width = LABEL_W + len(keys) * PANEL_W
    block_h = HEADER_H + rows_per_block * PANEL_H
    canvas = Image.new("RGB", (width, 2 * block_h + GAP), "white")
    draw = ImageDraw.Draw(canvas)
    for block in range(2):
        top = block * (block_h + GAP)
        for column, label in enumerate(labels):
            center_text(draw, LABEL_W + column * PANEL_W + PANEL_W // 2, top + 7, label, font(16, bold=True))
        for row, (title, panels) in enumerate(
            examples[block * rows_per_block : (block + 1) * rows_per_block]
        ):
            y = top + HEADER_H + row * PANEL_H
            draw.multiline_text((8, y + 42), title.replace(" - ", "\n"), fill="#202020", font=font(15, bold=True), spacing=2)
            for column, key in enumerate(keys):
                paste_bordered(canvas, panels[key], LABEL_W + column * PANEL_W, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def fixed_local_crop(panel: Image.Image) -> Image.Image:
    # The same lower-center normalized crop is used for every example and panel.
    left = int(panel.width * 0.25)
    top = int(panel.height * 0.36)
    right = int(panel.width * 0.75)
    bottom = panel.height
    return panel.crop((left, top, right, bottom)).resize((256, 160), Image.Resampling.NEAREST)


def assemble_s4(primary: Image.Image, hardest: Image.Image, output: Path) -> None:
    examples = example_panels(primary, hardest)
    keys = ("input", "ground_truth", "selected", "error")
    labels = ("Input crop", "Ground-truth crop", "Selected crop", "Error crop")
    crop_w, crop_h = 256, 160
    width = LABEL_W + len(keys) * crop_w
    height = HEADER_H + len(examples) * crop_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        center_text(draw, LABEL_W + column * crop_w + crop_w // 2, 7, label, font(16, bold=True))
    for row, (title, panels) in enumerate(examples):
        y = HEADER_H + row * crop_h
        draw.multiline_text((8, y + 52), title.replace(" - ", "\n"), fill="#202020", font=font(15, bold=True), spacing=2)
        for column, key in enumerate(keys):
            paste_bordered(canvas, fixed_local_crop(panels[key]), LABEL_W + column * crop_w, y)
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
    outputs = {
        "fig1": output_dir / "fig1_visual_domains_v11.png",
        "fig9": output_dir / "fig9_qualitative_capacity_v11.png",
        "figS1": output_dir / "figS1_expanded_failures_v11.png",
        "figS3": output_dir / "figS3_additional_qualitative_v11.png",
        "figS4": output_dir / "figS4_local_crops_v11.png",
    }
    assemble_fig1(primary, outputs["fig1"])
    assemble_fig9(primary, outputs["fig9"])
    assemble_s1(primary, hardest, outputs["figS1"])
    assemble_s3(primary, hardest, outputs["figS3"])
    assemble_s4(primary, hardest, outputs["figS4"])

    audit = {
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_hashes": {path.name: sha256(path) for path in (primary_path, hardest_path)},
        "evidence_boundary": (
            "V11 reorganizes the five audited median-error cases and one audited highest-error "
            "night case; it does not introduce new easy/hard selections."
        ),
        "local_crop_rule": "fixed normalized lower-center crop: x=[0.25,0.75], y=[0.36,1.00]",
        "outputs": {path.name: sha256(path) for path in outputs.values()},
    }
    audit_path = output_dir / "visual_story_v11_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for path in (*outputs.values(), audit_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
