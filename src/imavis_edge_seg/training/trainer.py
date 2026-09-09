"""Supernet training loop orchestration: optimizer, LR schedule, the sandwich-rule
step from `training.step`, periodic logging and checkpointing. Kept separate from
`scripts/train_supernet.py` so the loop itself is importable/testable without going
through the CLI.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterator
from pathlib import Path

import torch
from rich.console import Console
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from imavis_edge_seg.config import ExperimentConfig
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.training.checkpoint import save_checkpoint
from imavis_edge_seg.training.step import train_step

_Sample = tuple[torch.Tensor, torch.Tensor]


def _lr_lambda(step: int, config: ExperimentConfig) -> float:
    warmup = max(config.training.warmup_steps, 1)
    if step < warmup:
        return float(step) / float(warmup)
    progress = float(step - warmup) / float(max(config.training.max_steps - warmup, 1))
    progress = min(progress, 1.0)
    if config.training.lr_schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if config.training.lr_schedule == "poly":
        return float((1.0 - progress) ** 0.9)
    return 1.0  # constant


def _infinite_batches(loader: DataLoader[_Sample]) -> Iterator[_Sample]:
    # DataLoader(shuffle=True) draws a fresh shuffle order from the global torch RNG on
    # every `for batch in loader` pass, so simply looping forever already reshuffles
    # each "epoch" -- no manual reseeding needed.
    for _ in itertools.count():
        yield from loader


def run_training(
    config: ExperimentConfig,
    dataloader: DataLoader[_Sample],
    output_dir: Path,
    console: Console | None = None,
    device: str = "cpu",
) -> PaceSegSupernet:
    console = console or Console()
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)

    supernet = PaceSegSupernet(config.supernet).to(device)
    optimizer = AdamW(
        supernet.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: _lr_lambda(step, config))

    batches = _infinite_batches(dataloader)
    running_loss = 0.0

    for step in range(1, config.training.max_steps + 1):
        image, mask = next(batches)
        image = image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        result = train_step(supernet, image, mask, config, rng)
        result.total_loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap, not ours
        torch.nn.utils.clip_grad_norm_(supernet.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        running_loss += float(result.total_loss.detach())

        if step % config.training.log_interval_steps == 0:
            avg_loss = running_loss / config.training.log_interval_steps
            running_loss = 0.0
            lr = scheduler.get_last_lr()[0]
            console.print(
                f"step {step}/{config.training.max_steps} "
                f"loss={avg_loss:.4f} lr={lr:.2e} levels={result.levels_trained}"
            )

        if step % config.training.checkpoint_interval_steps == 0 or step == config.training.max_steps:
            ckpt_path = output_dir / "checkpoints" / f"step_{step:08d}.pt"
            save_checkpoint(ckpt_path, supernet, optimizer, step, config)
            console.print(f"checkpoint: {ckpt_path}")

    return supernet
