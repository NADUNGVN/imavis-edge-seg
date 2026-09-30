"""Render evidence-backed dataset and qualitative segmentation figures.

Run this on a machine that has the licensed Cityscapes/ACDC trees and the recorded
Run A checkpoint.  The selection itself is frozen by ``selection_manifest.json``.
Before rendering, every source checksum and every regenerated confusion matrix is
checked against the canonical deployment-matched per-image artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from imavis_edge_seg.config import ElasticityLevel, RouterConfig, load_config
from imavis_edge_seg.data.labels import IGNORE_INDEX, id_mask_to_train_id
from imavis_edge_seg.data.transforms import SegmentationResizeToTensor
from imavis_edge_seg.evaluation.metrics import compute_confusion_matrix
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.router.calibrator import RiskCalibrator
from imavis_edge_seg.router.policy import select_level
from imavis_edge_seg.search.pareto import ParetoPoint
from imavis_edge_seg.training.checkpoint import load_checkpoint

LEVELS: list[ElasticityLevel] = ["tiny", "small", "medium", "large"]
PALETTE = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _colorize(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    rgb[valid] = PALETTE[mask[valid]]
    return Image.fromarray(rgb, mode="RGB")


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def _fit_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return mask.resize(size, Image.Resampling.NEAREST)


def _error_overlay(
    image: Image.Image, prediction: np.ndarray, target: np.ndarray, size: tuple[int, int]
) -> Image.Image:
    base = np.asarray(_fit_panel(image, size), dtype=np.float32) * 0.45
    pred = np.asarray(_fit_mask(Image.fromarray(prediction.astype(np.uint8)), size))
    gt = np.asarray(_fit_mask(Image.fromarray(target.astype(np.uint8)), size))
    mismatch = (gt != IGNORE_INDEX) & (pred != gt)
    base[mismatch] = 0.25 * base[mismatch] + 0.75 * np.asarray((213, 94, 0))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def _label_panel(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result, "RGBA")
    bbox = draw.textbbox((0, 0), label, font=font)
    width = bbox[2] - bbox[0] + 12
    draw.rectangle((0, 0, width, bbox[3] - bbox[1] + 10), fill=(0, 0, 0, 180))
    draw.text((6, 4), label, fill=(255, 255, 255, 255), font=font)
    return result


def _load_target(entry: dict[str, Any], label_path: Path) -> np.ndarray:
    raw = np.asarray(Image.open(label_path), dtype=np.uint8)
    return id_mask_to_train_id(raw) if entry["split"] == "cityscapes" else raw


def _root_for(entry: dict[str, Any], cityscapes_root: Path, acdc_root: Path) -> Path:
    return cityscapes_root if entry["split"] == "cityscapes" else acdc_root


def _select_d(
    split_dump: dict[str, Any],
    split_replay: dict[str, Any],
    heldout_position: int,
    budget_index: int,
) -> tuple[ElasticityLevel, float, float, dict[ElasticityLevel, float]]:
    route_costs = split_replay["_budget_grid_ms"]
    budget = float(route_costs[budget_index])
    points = split_replay["e2e_aware"]["risk_latency_constrained"]
    budget_key = min(points, key=lambda key: abs(float(key) - budget))
    risk_target = float(points[budget_key]["risk_target"])
    calibrators = {
        level: RiskCalibrator.from_dict(split_dump["per_level_calibrators"][level])
        for level in LEVELS
    }
    raw_score = float(split_dump["test_raw_scores"][heldout_position])
    predicted_risks = {level: calibrators[level].predict(raw_score) for level in LEVELS}
    candidates = [
        ParetoPoint(
            level=level,
            device_id="qualitative",
            backend="measured_e2e",
            precision="fp16",
            latency_ms=float(route_costs[index]),
            miou=0.0,
            dataset="n/a",
        )
        for index, level in enumerate(LEVELS)
    ]
    config = RouterConfig(
        strategy="risk_latency_constrained",
        risk_target=risk_target,
        latency_budget_ms=budget,
    )
    selected = select_level(candidates, config, per_level_risk=predicted_risks)
    return selected, budget, risk_target, predicted_risks


@torch.no_grad()
def _render_entry(
    entry: dict[str, Any],
    model: PaceSegSupernet,
    config: Any,
    device: str,
    cityscapes_root: Path,
    acdc_root: Path,
    dump: dict[str, Any],
    replay: dict[str, Any],
    budget_index: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    root = _root_for(entry, cityscapes_root, acdc_root)
    image_path = root / entry["image_path"]
    label_path = root / entry["label_path"]
    if _sha256(image_path) != entry["image_sha256"]:
        raise ValueError(f"image checksum mismatch: {image_path}")
    if _sha256(label_path) != entry["label_sha256"]:
        raise ValueError(f"label checksum mismatch: {label_path}")

    raw_image = Image.open(image_path).convert("RGB")
    target = _load_target(entry, label_path)
    predictions: dict[ElasticityLevel, np.ndarray] = {}
    for level in LEVELS:
        height, width = config.supernet.input_resolutions[level]
        transform = SegmentationResizeToTensor(height=height, width=width)
        image_tensor, mask_tensor = transform(raw_image, Image.fromarray(target))
        logits = model(image_tensor.unsqueeze(0).to(device), level)
        prediction = logits.argmax(dim=1).cpu()
        expected = dump[entry["split"]]["test_confusion_matrices"][level][
            entry["heldout_position"]
        ]
        actual = compute_confusion_matrix(
            prediction, mask_tensor.unsqueeze(0), config.supernet.num_classes
        ).tolist()
        if actual != expected:
            raise ValueError(
                f"regenerated prediction does not match canonical artifact: "
                f"{entry['image_id']} / {level}"
            )
        predictions[level] = prediction.squeeze(0).numpy().astype(np.uint8)

    replay_view = dict(replay[entry["split"]])
    replay_view["_budget_grid_ms"] = replay["_metadata"]["budget_grid_ms"]
    selected, budget, risk_target, predicted_risks = _select_d(
        dump[entry["split"]], replay_view, entry["heldout_position"], budget_index
    )

    panel_size = (256, 128)
    target_panel = _colorize(target)
    panels = [
        _fit_panel(raw_image, panel_size),
        _fit_mask(target_panel, panel_size),
        *[_fit_mask(_colorize(predictions[level]), panel_size) for level in LEVELS],
        _fit_mask(_colorize(predictions[selected]), panel_size),
        _error_overlay(raw_image, predictions[selected], target, panel_size),
    ]
    result = {
        **entry,
        "selected_candidate": selected,
        "budget_ms": budget,
        "risk_target": risk_target,
        "predicted_risk": predicted_risks,
        "confusion_matrix_audit": "exact match for all four levels",
    }
    return panels, result


def _assemble_grid(
    rows: list[tuple[list[Image.Image], dict[str, Any]]],
    headers: list[str],
    output: Path,
) -> None:
    font = _font(18)
    header_font = _font(20)
    panel_w, panel_h = rows[0][0][0].size
    header_h = 34
    canvas = Image.new("RGB", (panel_w * len(headers), header_h + panel_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        bbox = draw.textbbox((0, 0), header, font=header_font)
        x = column * panel_w + (panel_w - (bbox[2] - bbox[0])) // 2
        draw.text((x, 5), header, fill=(25, 25, 25), font=header_font)
    for row_index, (panels, metadata) in enumerate(rows):
        for column, panel in enumerate(panels):
            label = ""
            if column == 0:
                label = metadata["condition"].title()
            elif column == 6:
                label = f"{metadata['selected_candidate']} | B={metadata['budget_ms']:.1f} ms"
            elif column == 7:
                label = "error"
            if label:
                panel = _label_panel(panel, label, font)
            canvas.paste(panel, (column * panel_w, header_h + row_index * panel_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def _assemble_dataset_overview(
    rows: list[tuple[list[Image.Image], dict[str, Any]]], output: Path
) -> None:
    font = _font(20)
    panel_w, panel_h = rows[0][0][0].size
    header_h = 34
    canvas = Image.new("RGB", (panel_w * len(rows), header_h + panel_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (panels, metadata) in enumerate(rows):
        label = metadata["condition"].title()
        bbox = draw.textbbox((0, 0), label, font=font)
        x = column * panel_w + (panel_w - (bbox[2] - bbox[0])) // 2
        draw.text((x, 5), label, fill=(25, 25, 25), font=font)
        canvas.paste(panels[0], (column * panel_w, header_h))
        canvas.paste(_label_panel(panels[1], "ground truth", _font(17)), (column * panel_w, header_h + panel_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, dpi=(300, 300), optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=Path("paper/figures/qualitative/selection_manifest.json"),
    )
    parser.add_argument(
        "--per-image-dump",
        type=Path,
        default=Path("reports/router_deployment_matched_20260929/run_a_per_image.json"),
    )
    parser.add_argument(
        "--replay-json",
        type=Path,
        default=Path("reports/router_deployment_matched_20260929/run_a_e3_replay.json"),
    )
    parser.add_argument("--cityscapes-root", type=Path)
    parser.add_argument("--acdc-root", type=Path)
    parser.add_argument("--budget-index", type=int, default=2, choices=range(4))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper/figures/qualitative/generated")
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with args.selection_manifest.open(encoding="utf-8") as stream:
        selection = json.load(stream)
    with args.per_image_dump.open(encoding="utf-8") as stream:
        dump = json.load(stream)
    with args.replay_json.open(encoding="utf-8") as stream:
        replay = json.load(stream)
    if _sha256(args.checkpoint) != selection["source_checkpoint_sha256"]:
        raise ValueError("checkpoint SHA-256 does not match the frozen selection manifest")

    config = load_config(args.config)
    roots = {item.name: Path(item.root) for item in config.datasets}
    cityscapes_root = args.cityscapes_root or roots["cityscapes"]
    acdc_root = args.acdc_root or roots["acdc"]
    model = PaceSegSupernet(config.supernet).to(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rendered = [
        _render_entry(
            entry,
            model,
            config,
            args.device,
            cityscapes_root,
            acdc_root,
            dump,
            replay,
            args.budget_index,
        )
        for entry in selection["selections"]
    ]
    primary = [row for row in rendered if row[1]["selection_kind"] == "median_large_error"]
    failure = [row for row in rendered if row[1]["selection_kind"].startswith("hardest_")]
    headers = ["Input", "Ground truth", "Tiny", "Small", "Medium", "Large", "PACE-Seg", "Error"]
    dataset_path = args.output_dir / "fig6_dataset_overview.png"
    qualitative_path = args.output_dir / "fig7_qualitative_grid.png"
    failure_path = args.output_dir / "figS1_hardest_failure.png"
    _assemble_dataset_overview(primary, dataset_path)
    _assemble_grid(primary, headers, qualitative_path)
    _assemble_grid(failure, headers, failure_path)

    render_manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "per_image_dump": str(args.per_image_dump),
        "replay_json": str(args.replay_json),
        "budget_index": args.budget_index,
        "examples": [metadata for _, metadata in rendered],
        "outputs": {
            path.name: _sha256(path)
            for path in (dataset_path, qualitative_path, failure_path)
        },
    }
    manifest_path = args.output_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(render_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {dataset_path}")
    print(f"wrote {qualitative_path}")
    print(f"wrote {failure_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
