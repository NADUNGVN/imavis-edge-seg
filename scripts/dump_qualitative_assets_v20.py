"""Dump the image assets for the V20 qualitative figures (run once on the server).

Inputs: paper/figures/qualitative/selection_v20.json (frozen locally by
paper/figures/scripts/select_qualitative_v20.py), the landscape Run A checkpoint,
and the licensed Cityscapes/ACDC trees named in the config.

For every selected held-out image it writes, under --out:
  <id>_rgb.jpg            input resized to 1024x512 (JPEG q92)
  <id>_gt.png             19-class train-id label at 1024x512 (255 = ignore)
  <id>_<level>.png        argmax prediction of each capacity, upsampled to 1024x512 (nearest)
  <id>_entropy.npy        tiny-candidate per-pixel entropy (float16, tiny resolution)
and manifest.json with the per-image probe score. Every prediction is audited
against the confusion matrices of the canonical per-image dump before writing, so
the figures show exactly the predictions that the paper's numbers come from.
Rendering happens locally from these files; nothing here draws a figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imavis_edge_seg.config import load_config
from imavis_edge_seg.data.labels import id_mask_to_train_id
from imavis_edge_seg.data.transforms import SegmentationResizeToTensor
from imavis_edge_seg.evaluation.metrics import compute_confusion_matrix
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.training.checkpoint import load_checkpoint

LEVELS = ("tiny", "small", "medium", "large")
SIZE = (1024, 512)  # width, height of every dumped panel


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def image_id(entry: dict) -> str:
    split = entry["split"].replace("/", "-")
    return f"{split}_{Path(entry['image_path']).stem}"


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=Path("paper/figures/qualitative/selection_v20.json"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("outputs/pace_seg_landscape_supernet_seed0/checkpoints/step_00100000.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    p.add_argument("--per-image-dump", type=Path, default=Path("reports/landscape_20261004/run_a_per_image.json"))
    p.add_argument("--out", type=Path, default=Path("reports/qualitative_assets_v20"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sel = json.loads(args.selection.read_text())
    if sha256(args.checkpoint) != sel["source_checkpoint_sha256"]:
        raise SystemExit("checkpoint SHA-256 does not match the selection manifest")
    dump = json.loads(args.per_image_dump.read_text())
    config = load_config(args.config)
    roots = {d.name: Path(d.root) for d in config.datasets}
    model = PaceSegSupernet(config.supernet).to(args.device).eval()
    model.load_state_dict(load_checkpoint(args.checkpoint, map_location=args.device)["model_state_dict"])
    args.out.mkdir(parents=True, exist_ok=True)

    manifest, done = [], set()
    for entry in sel["selections"]:
        iid = image_id(entry)
        record = {**entry, "asset_id": iid}
        manifest.append(record)
        if iid in done:
            continue
        done.add(iid)
        root = roots["cityscapes" if entry["split"] == "cityscapes" else "acdc"]
        img_path, lbl_path = root / entry["image_path"], root / entry["label_path"]
        if sha256(img_path) != entry["image_sha256"] or sha256(lbl_path) != entry["label_sha256"]:
            raise SystemExit(f"dataset checksum mismatch: {img_path}")
        rgb = Image.open(img_path).convert("RGB")
        raw = np.asarray(Image.open(lbl_path), dtype=np.uint8)
        target = id_mask_to_train_id(raw) if entry["split"] == "cityscapes" else raw
        for level in LEVELS:
            h, w = config.supernet.input_resolutions[level]
            x, m = SegmentationResizeToTensor(height=h, width=w)(rgb, Image.fromarray(target))
            logits = model(x.unsqueeze(0).to(args.device), level)
            pred = logits.argmax(1).cpu()
            cm = compute_confusion_matrix(pred, m.unsqueeze(0), config.supernet.num_classes).tolist()
            if cm != dump[entry["split"]]["test_confusion_matrices"][level][entry["heldout_position"]]:
                raise SystemExit(f"prediction does not match the canonical dump: {iid}/{level}")
            Image.fromarray(pred[0].numpy().astype(np.uint8)).resize(SIZE, Image.Resampling.NEAREST) \
                .save(args.out / f"{iid}_{level}.png", optimize=True)
            if level == "tiny":
                prob = torch.softmax(logits.float(), 1)[0]
                ent = -(prob * prob.clamp_min(1e-12).log()).sum(0)
                np.save(args.out / f"{iid}_entropy.npy", ent.cpu().numpy().astype(np.float16))
                record["probe_score"] = float(ent.mean())
        rgb.resize(SIZE, Image.Resampling.BILINEAR).save(args.out / f"{iid}_rgb.jpg", quality=92)
        Image.fromarray(target).resize(SIZE, Image.Resampling.NEAREST).save(args.out / f"{iid}_gt.png", optimize=True)
        print(f"ok {iid}  score={record['probe_score']:.4f}", flush=True)
    for r in manifest:  # duplicates (same image picked twice) share the first record's score
        r.setdefault("probe_score", next(x["probe_score"] for x in manifest if x["asset_id"] == r["asset_id"] and "probe_score" in x))
    (args.out / "manifest.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sel["source_checkpoint_sha256"],
        "panel_size_wh": SIZE, "audit": "all predictions match the canonical per-image confusion matrices",
        "entries": manifest}, indent=2))
    print(f"wrote {len(done)} images to {args.out}")


if __name__ == "__main__":
    main()
