"""Optional training-speed settings shared by the supernet and baseline trainers.

All are off by default so earlier runs stay reproducible; the 2026-10-03 landscape
retraining enables them through config overrides. `amp` uses FP16 autocast with a
GradScaler (Turing RTX 8000 has no BF16); softmax/log-softmax/cross-entropy run in
FP32 under autocast, so the losses keep FP32 numerics.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext

import torch
from torch import nn
from torch.optim import Optimizer

from imavis_edge_seg.config import ExperimentConfig


class SpeedSettings:
    def __init__(self, config: ExperimentConfig, device: str, qat: bool) -> None:
        cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.amp = bool(config.training.amp) and cuda
        if self.amp and qat:
            raise ValueError("training.amp is not supported together with QAT fake quantization")
        if config.training.cudnn_benchmark and cuda:
            torch.backends.cudnn.benchmark = True
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self._t0 = time.perf_counter()
        self._last_step: int | None = None

    def autocast(self) -> AbstractContextManager[object]:
        if self.amp:
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def backward_and_step(self, loss: torch.Tensor, model: nn.Module, optimizer: Optimizer, clip: float) -> None:
        self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        self.scaler.step(optimizer)
        self.scaler.update()

    def throughput(self, step: int, max_steps: int) -> str:
        """Steps/s and ETA since the previous call (call at log intervals)."""
        now = time.perf_counter()
        if self._last_step is None:
            self._last_step, self._t0 = step, now
            return "speed=warming-up"
        rate = (step - self._last_step) / max(now - self._t0, 1e-9)
        self._last_step, self._t0 = step, now
        eta_h = (max_steps - step) / max(rate, 1e-9) / 3600
        return f"speed={rate:.2f}it/s eta={eta_h:.2f}h"
