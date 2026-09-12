"""Supernet training loop orchestration: optimizer, LR schedule, the sandwich-rule
step from `training.step`, periodic logging and checkpointing. Kept separate from
`scripts/train_supernet.py` so the loop itself is importable/testable without going
through the CLI.
"""

from __future__ import annotations

import itertools
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
from imavis_edge_seg.training.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from imavis_edge_seg.training.schedule import lr_lambda
from imavis_edge_seg.training.step import train_step

_Sample = tuple[torch.Tensor, torch.Tensor]


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
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, config))

    # Resume if a checkpoint already exists for this experiment_id (e.g. after an
    # accidental kill or a shared-server contention crash) -- without this, any
    # interruption loses all progress since the *previous* run's completion, not just
    # since the last checkpoint write. `rng`'s sampled-level sequence still restarts
    # from `config.seed` on resume (not bit-exact vs. an uninterrupted run), but model/
    # optimizer/step state all continue correctly.
    start_step = 0
    latest = find_latest_checkpoint(output_dir / "checkpoints")
    if latest is not None:
        checkpoint = load_checkpoint(latest, map_location=device)
        if checkpoint["config_hash"] != config.config_hash():
            console.print(
                f"[yellow]found {latest} but its config_hash {checkpoint['config_hash']!r} != "
                f"current config's {config.config_hash()!r} -- ignoring it and starting from "
                "scratch (architecture/hyperparameters changed since that checkpoint)[/yellow]"
            )
        else:
            supernet.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint["step"]
            for _ in range(start_step):
                scheduler.step()
            console.print(f"resumed from {latest} at step {start_step}")

    if start_step >= config.training.max_steps:
        console.print(
            f"checkpoint already at step {start_step} >= max_steps "
            f"{config.training.max_steps} -- nothing to do"
        )
        return supernet

    batches = _infinite_batches(dataloader)
    running_loss = 0.0

    for step in range(start_step + 1, config.training.max_steps + 1):
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
