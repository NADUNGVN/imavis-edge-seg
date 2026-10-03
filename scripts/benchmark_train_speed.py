"""Measure training throughput per server before the 2026-10-03 landscape retraining.

For each job type (supernet, fast_scnn, pace_large) and each setting
(num_workers x amp), runs `--warmup` + `--steps` real training steps on the real
Cityscapes+ACDC training stream (landscape resolutions) and reports steps/s, the
projected time for 100k steps, peak GPU memory, and the pure data-loader rate (so a
CPU/IO-bound setting is visible). No checkpoints, no git writes.

  python scripts/benchmark_train_speed.py --output reports/server/<host>_train_speed_<utc>.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import random
import socket
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from imavis_edge_seg.config import load_config
from imavis_edge_seg.models.baselines import build_baseline_model
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.training.data import build_train_dataloader
from imavis_edge_seg.training.losses import boundary_aware_segmentation_loss
from imavis_edge_seg.training.speed import SpeedSettings
from imavis_edge_seg.training.step import train_step


def measure(job: str, workers: int, amp: bool, args: argparse.Namespace) -> dict:
    config = load_config(args.config, overrides=[f"training.num_workers={workers}", f"training.amp={amp}",
                                                  "training.cudnn_benchmark=true", "seed=0"])
    loader = build_train_dataloader(config)
    it = iter(loader)
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    if job == "supernet":
        model = PaceSegSupernet(config.supernet).to(device)
    else:
        model = build_baseline_model(job, config.supernet.num_classes).to(device)
    opt = AdamW(model.parameters(), lr=config.training.lr)
    speed = SpeedSettings(config, device, qat=False)
    rng = random.Random(0)

    def one_step() -> None:
        image, mask = next(it)
        image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with speed.autocast():
            if job == "supernet":
                loss = train_step(model, image, mask, config, rng).total_loss  # type: ignore[arg-type]
            else:
                loss = boundary_aware_segmentation_loss(model(image), mask, config.training.boundary_loss_weight)
        speed.backward_and_step(loss, model, opt, config.training.grad_clip_norm)

    for _ in range(args.warmup):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()
    rate = args.steps / (time.perf_counter() - t0)

    # pure data-loader rate with the same workers (batches/s), model idle
    t0 = time.perf_counter()
    for _ in range(args.steps):
        next(it)
    data_rate = args.steps / (time.perf_counter() - t0)
    out = {"job": job, "num_workers": workers, "amp": amp, "steps_per_s": rate,
           "hours_per_100k": 100_000 / rate / 3600, "data_batches_per_s": data_rate,
           "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30}
    del it, loader, model, opt
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    ap.add_argument("--jobs", nargs="+", default=["supernet", "fast_scnn", "pace_large"])
    ap.add_argument("--workers", nargs="+", type=int, default=[4, 8, 12])
    ap.add_argument("--amp", nargs="+", default=["false", "true"])
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {"host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "cpu_count": __import__("os").cpu_count(), "platform": platform.platform(), "runs": []}
    print(f"{result['host']} {result['gpu']} torch {result['torch']}", flush=True)
    for job, workers, amp in itertools.product(args.jobs, args.workers, [a == "true" for a in args.amp]):
        try:
            r = measure(job, workers, amp, args)
        except Exception as exc:  # recorded, never hidden
            r = {"job": job, "num_workers": workers, "amp": amp, "error": repr(exc)[:300]}
        result["runs"].append(r)
        print(json.dumps(r), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
