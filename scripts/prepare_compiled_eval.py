"""Prepare trained-weight ONNX graphs and an evaluation bundle for compiled-engine
accuracy on E1/E3 (review round 2, 2026-10-03). Run ON THE GPU SERVER.

Outputs (all under --out-dir, which must be OUTSIDE the git worktree -- ONNX files,
images and labels are never committed):
  onnx/pace_seg_<level>.onnx            trained Run-A static subnets (opset 17, batch 1)
  eval/<split>__<level>.npz             held-out (odd-index) samples of that split at the
                                        level's input resolution:
                                          images  uint8 (N,H,W,3)  resized RGB, NOT normalized
                                          labels  uint8 (N,H,W)    trainIds, 255 = ignore
                                          torch_pred uint8 (N,H,W) PyTorch argmax reference
                                          indices int (N,)         dataset index
                                        tiny files also store torch_entropy float32 (N,)
  calib/<level>.npz                     first --calib-per-split fit-half (even-index)
                                        images per split, uint8, for Hailo INT8 calibration
  summary.json                          PyTorch held-out mIoU per split/level, file hashes
Images are produced by the exact PIL resize of SegmentationResizeToTensor; the script
asserts that re-normalizing the stored uint8 image reproduces the loader tensor.

Prints a human-readable summary (captured into the server report by the wrapper).
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
from imavis_edge_seg.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from imavis_edge_seg.evaluation.data import build_acdc_eval_loader, build_cityscapes_eval_loader
from imavis_edge_seg.export import export_subnet_onnx
from imavis_edge_seg.models.subnet import extract_subnet
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.router.risk_probe import compute_deployment_risk_score
from imavis_edge_seg.training.checkpoint import load_checkpoint

SPLITS = ("cityscapes", "acdc/fog", "acdc/night", "acdc/rain", "acdc/snow")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def miou(conf: np.ndarray) -> float:
    tp = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou))


def resized_uint8(path: Path, height: int, width: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def normalize(u8: np.ndarray) -> np.ndarray:
    x = u8.astype(np.float32) / 255.0
    for c in range(3):
        x[:, :, c] = (x[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return x.transpose(2, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--calib-per-split", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = args.out_dir.resolve()
    if (Path.cwd() / ".git").exists() and str(out).startswith(str(Path.cwd().resolve())):
        raise SystemExit("--out-dir must be outside the git worktree")
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(exist_ok=True)
    (out / "calib").mkdir(exist_ok=True)

    config = load_config(args.config)
    supernet = PaceSegSupernet(config.supernet).to(args.device)
    ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    supernet.load_state_dict(ckpt["model_state_dict"])
    supernet.eval()
    levels = list(config.supernet.levels)
    roots = {d.name: d.root for d in config.datasets}
    summary: dict = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                     "checkpoint_step": ckpt.get("step"), "config_hash": ckpt.get("config_hash"),
                     "torch_heldout_miou": {}, "files": {}}
    print(f"checkpoint step={ckpt.get('step')} sha256={summary['checkpoint_sha256'][:12]}")

    for level in levels:
        h, w = config.supernet.input_resolutions[level]
        sub = extract_subnet(supernet, level).cpu().eval()
        path = export_subnet_onnx(sub, out / "onnx" / f"pace_seg_{level}.onnx", h, w)
        summary["files"][str(path.relative_to(out))] = sha256(path)
        print(f"exported {path.name} input 1x3x{h}x{w}")

    calib: dict[str, list[np.ndarray]] = {lv: [] for lv in levels}
    with torch.no_grad():
        for split in SPLITS:
            for level in levels:
                h, w = config.supernet.input_resolutions[level]
                if split == "cityscapes":
                    loader = build_cityscapes_eval_loader(config, level, roots["cityscapes"], "val", batch_size=1)
                else:
                    loader = build_acdc_eval_loader(config, level, roots["acdc"], split.split("/")[1], "val", batch_size=1)
                samples = loader.dataset.samples  # type: ignore[attr-defined]
                imgs, labels, preds, idxs, ents = [], [], [], [], []
                conf = np.zeros((19, 19), dtype=np.int64)
                n_calib = 0
                for index, (image, mask) in enumerate(loader):
                    u8 = resized_uint8(Path(samples[index][0]), h, w)
                    ref = image[0].numpy()
                    if not np.allclose(normalize(u8), ref, atol=1e-5):
                        raise SystemExit(f"stored uint8 image does not reproduce loader tensor: {split} {index} {level}")
                    if index % 2 == 0:
                        if n_calib < args.calib_per_split:
                            calib[level].append(u8)
                            n_calib += 1
                        continue
                    logits = supernet(image.to(args.device), level)
                    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    lab = mask[0].numpy()
                    lab = np.where((lab >= 0) & (lab <= 18), lab, 255).astype(np.uint8)
                    valid = lab != 255
                    conf += np.bincount(lab[valid].astype(int) * 19 + pred[valid], minlength=361).reshape(19, 19)
                    imgs.append(u8)
                    labels.append(lab)
                    preds.append(pred)
                    idxs.append(index)
                    if level == levels[0]:
                        ents.append(float(compute_deployment_risk_score(logits)))
                fname = out / "eval" / f"{split.replace('/', '_')}__{level}.npz"
                extra = {"torch_entropy": np.array(ents, np.float32)} if ents else {}
                np.savez_compressed(fname, images=np.stack(imgs), labels=np.stack(labels),
                                    torch_pred=np.stack(preds), indices=np.array(idxs), **extra)
                summary["torch_heldout_miou"][f"{split}|{level}"] = miou(conf)
                summary["files"][str(fname.relative_to(out))] = sha256(fname)
                print(f"{split:12s} {level:6s} held-out n={len(idxs)} torch mIoU={miou(conf) * 100:.2f}")
    for level, arr in calib.items():
        fname = out / "calib" / f"{level}.npz"
        np.savez_compressed(fname, images=np.stack(arr))
        summary["files"][str(fname.relative_to(out))] = sha256(fname)
        print(f"calib {level}: {len(arr)} images")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"bundle {out} total {total:.1f} MB")


if __name__ == "__main__":
    main()
