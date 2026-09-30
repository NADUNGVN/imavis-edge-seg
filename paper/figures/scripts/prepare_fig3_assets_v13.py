"""Prepare audited raster panels for the symbolic V13 routing schematic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

CONDITIONS = ("clean", "fog", "night", "rain", "snow")
COLUMNS = {"input": 0, "tiny": 2, "selected": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--condition", choices=CONDITIONS, default="rain")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parents[1] / "v13" / "tikz" / "assets",
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
    return composite.crop(
        (
            col * panel_w,
            header_h + row * panel_h,
            (col + 1) * panel_w,
            header_h + (row + 1) * panel_h,
        )
    )


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    source_dir = repo / "paper" / "figures" / "qualitative" / "generated"
    composite_path = source_dir / "fig7_qualitative_grid.png"
    manifest_path = source_dir / "render_manifest.json"
    manifest = read_json(manifest_path)
    expected_hash = manifest["outputs"][composite_path.name]
    if sha256(composite_path) != expected_hash:
        raise RuntimeError("Qualitative composite hash does not match render manifest")

    entry = next(item for item in manifest["examples"] if item["condition"] == args.condition)
    composite = Image.open(composite_path).convert("RGB")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "input": output_dir / "fig3_input.png",
        "probe": output_dir / "fig3_tiny_probe.png",
        "output": output_dir / "fig3_selected_output.png",
    }
    crop_panel(composite, args.condition, "input").save(outputs["input"], optimize=True)
    crop_panel(composite, args.condition, "tiny").save(outputs["probe"], optimize=True)
    crop_panel(composite, args.condition, "selected").save(outputs["output"], optimize=True)

    audit = {
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_composite": str(composite_path.relative_to(repo)),
        "source_composite_sha256": expected_hash,
        "condition": args.condition,
        "image_id": entry["image_id"],
        "evidence_boundary": "Unmodified audited raster crops; the schematic carries no numerical claim.",
        "outputs": {path.name: sha256(path) for path in outputs.values()},
    }
    audit_path = output_dir / "fig3_assets_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for path in (*outputs.values(), audit_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
