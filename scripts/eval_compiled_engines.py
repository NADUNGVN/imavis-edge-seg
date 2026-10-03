"""Accuracy of trained-weight compiled engines on the edge devices (review round 2,
2026-10-03). Runs ON THE DEVICE (E3 TensorRT, E1 Hailo-8).

Inputs: the evaluation bundle from scripts/prepare_compiled_eval.py
(eval/<split>__<level>.npz with uint8 images, trainId labels, PyTorch argmax and,
for tiny, PyTorch entropy), and the four compiled engines.

For every split and level it reports: compiled mIoU, PyTorch mIoU on the same images,
pixel agreement between compiled and PyTorch argmax, and for tiny the deviation of the
mean-entropy probe score from PyTorch. Per-image confusion matrices of the compiled
predictions are stored (npz) so the routing replay can be rerun with on-device
predictions.

Normalization is identical to SegmentationResizeToTensor (x/255, ImageNet mean/std).
TensorRT engines take NCHW float32 and return NCHW logits; Hailo HEFs take NHWC float32
through FLOAT32 vstreams (host quantization) and return NHWC float32 logits.

  E3: python eval_compiled_engines.py --backend trt --engine-dir trained_engines --eval-dir compiled_eval_20261003/eval --output-prefix compiled_eval_E3
  E1: python3 eval_compiled_engines.py --backend hailo --engine-dir trained_hef --eval-dir compiled_eval_20261003/eval --output-prefix compiled_eval_E1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

LEVELS = ("tiny", "small", "medium", "large")
SPLITS = ("cityscapes", "acdc_fog", "acdc_night", "acdc_rain", "acdc_snow")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalize_nhwc(u8: np.ndarray) -> np.ndarray:
    return ((u8.astype(np.float32) / 255.0) - MEAN) / STD


def miou(conf: np.ndarray) -> float:
    tp = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou))


def confusion(lab: np.ndarray, pred: np.ndarray) -> np.ndarray:
    valid = lab != 255
    return np.bincount(lab[valid].astype(int) * 19 + pred[valid].astype(int), minlength=361).reshape(19, 19)


def entropy_chw(logits: np.ndarray) -> float:
    x = logits - logits.max(axis=0, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=0, keepdims=True)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum(axis=0).mean())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


class TrtRunner:
    def __init__(self, path: Path) -> None:
        from measure_router_overhead_v2lib import TrtEngine  # E3 helper module

        self.engine = TrtEngine(path)

    def __call__(self, nhwc: np.ndarray) -> np.ndarray:
        flat = self.engine.infer_sync(np.ascontiguousarray(nhwc.transpose(2, 0, 1)[None]))
        return np.asarray(flat).reshape(self.engine.output_shape)[0]  # (C,H,W)


class HailoRunner:
    def __init__(self, path: Path, vdevice) -> None:
        import hailo_platform as hp

        self.hef = hp.HEF(str(path))
        params = hp.ConfigureParams.create_from_hef(self.hef, interface=hp.HailoStreamInterface.PCIe)
        self.ng = vdevice.configure(self.hef, params)[0]
        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_info = self.hef.get_output_vstream_infos()[0]
        ip = hp.InputVStreamParams.make(self.ng, format_type=hp.FormatType.FLOAT32)
        op = hp.OutputVStreamParams.make(self.ng, format_type=hp.FormatType.FLOAT32)
        self._cm = hp.InferVStreams(self.ng, ip, op)
        self.pipe = self._cm.__enter__()
        self._act = self.ng.activate(self.ng.create_params())
        self._act.__enter__()

    def __call__(self, nhwc: np.ndarray) -> np.ndarray:
        out = self.pipe.infer({self.in_info.name: nhwc[None]})[self.out_info.name][0]  # (H,W,C)
        return out.transpose(2, 0, 1)

    def close(self) -> None:
        self._act.__exit__(None, None, None)
        self._cm.__exit__(None, None, None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["trt", "hailo"], required=True)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    report: dict = {"backend": args.backend, "results": {}, "entropy": {}}
    per_image: dict[str, np.ndarray] = {}
    vdevice = None
    if args.backend == "hailo":
        import hailo_platform as hp

        vdevice = hp.VDevice()
    try:
        for level in LEVELS:
            ext = "engine" if args.backend == "trt" else "hef"
            path = args.engine_dir / f"pace_seg_{level}.{ext}"
            runner = TrtRunner(path) if args.backend == "trt" else HailoRunner(path, vdevice)
            for split in SPLITS:
                d = np.load(args.eval_dir / f"{split}__{level}.npz")
                imgs, labels, tpred = d["images"], d["labels"], d["torch_pred"]
                conf_c = np.zeros((19, 19), np.int64)
                conf_t = np.zeros((19, 19), np.int64)
                agree = total = 0
                cms, ents = [], []
                t0 = time.time()
                for i in range(len(imgs)):
                    logits = runner(normalize_nhwc(imgs[i]))
                    if logits.shape[1:] != labels[i].shape:
                        raise SystemExit(f"output {logits.shape} does not match label {labels[i].shape} ({split} {level})")
                    pred = logits.argmax(0).astype(np.uint8)
                    cm = confusion(labels[i], pred)
                    cms.append(cm)
                    conf_c += cm
                    conf_t += confusion(labels[i], tpred[i])
                    agree += int((pred == tpred[i]).sum())
                    total += pred.size
                    if level == "tiny":
                        ents.append(entropy_chw(logits))
                key = f"{split}|{level}"
                report["results"][key] = {"n": len(imgs), "compiled_miou": miou(conf_c), "torch_miou": miou(conf_t),
                                          "delta_points": (miou(conf_c) - miou(conf_t)) * 100,
                                          "pixel_agreement": agree / total, "seconds": time.time() - t0}
                per_image[key.replace("|", "__")] = np.stack(cms).astype(np.int32)
                if level == "tiny":
                    te = d["torch_entropy"]
                    ce = np.array(ents, np.float32)
                    report["entropy"][split] = {"mae": float(np.abs(ce - te).mean()), "max_abs": float(np.abs(ce - te).max()),
                                                "mean_torch": float(te.mean()), "mean_compiled": float(ce.mean()),
                                                "spearman": spearman(ce, te)}
                    per_image[f"{split}__tiny_entropy"] = ce
                r = report["results"][key]
                print(f"{split:11s} {level:6s} compiled {r['compiled_miou'] * 100:6.2f}  torch {r['torch_miou'] * 100:6.2f}  "
                      f"delta {r['delta_points']:+6.2f}  agree {r['pixel_agreement'] * 100:6.2f}%", flush=True)
            if args.backend == "hailo":
                runner.close()
    finally:
        if vdevice is not None:
            vdevice.release()
    Path(f"{args.output_prefix}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(f"{args.output_prefix}_per_image.npz", **per_image)
    print(f"wrote {args.output_prefix}.json and {args.output_prefix}_per_image.npz")


if __name__ == "__main__":
    main()
