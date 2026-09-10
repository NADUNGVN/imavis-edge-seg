"""Training loop for a single fixed-architecture required baseline (`RESEARCH_PLAN.md`
§7) -- plain segmentation + boundary-aware loss at the supernet's largest configured
resolution, no elasticity sampling, no in-place distillation. Deliberately mirrors
`training.trainer.run_training`'s structure (same optimizer/schedule/checkpoint
conventions) so results are comparable and reviewers see one consistent training
recipe, not two different ones invented for baselines vs. the proposed method.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path

import torch
from rich.console import Console
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from imavis_edge_seg.config import ExperimentConfig
from imavis_edge_seg.models.baselines import build_baseline_model
from imavis_edge_seg.training.checkpoint import save_checkpoint
from imavis_edge_seg.training.losses import boundary_aware_segmentation_loss
from imavis_edge_seg.training.schedule import lr_lambda

_Sample = tuple[torch.Tensor, torch.Tensor]


def _infinite_batches(loader: DataLoader[_Sample]) -> Iterator[_Sample]:
    for _ in itertools.count():
        yield from loader


def run_baseline_training(
    baseline_name: str,
    config: ExperimentConfig,
    dataloader: DataLoader[_Sample],
    output_dir: Path,
    console: Console | None = None,
    device: str = "cpu",
) -> nn.Module:
    console = console or Console()
    torch.manual_seed(config.seed)

    model = build_baseline_model(baseline_name, num_classes=config.supernet.num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, config))

    batches = _infinite_batches(dataloader)
    running_loss = 0.0

    for step in range(1, config.training.max_steps + 1):
        image, mask = next(batches)
        image = image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = boundary_aware_segmentation_loss(logits, mask, boundary_weight=config.training.boundary_loss_weight)
        loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap, not ours
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        running_loss += float(loss.detach())

        if step % config.training.log_interval_steps == 0:
            avg_loss = running_loss / config.training.log_interval_steps
            running_loss = 0.0
            lr = scheduler.get_last_lr()[0]
            console.print(f"[{baseline_name}] step {step}/{config.training.max_steps} loss={avg_loss:.4f} lr={lr:.2e}")

        if step % config.training.checkpoint_interval_steps == 0 or step == config.training.max_steps:
            ckpt_path = output_dir / "checkpoints" / f"step_{step:08d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, config)
            console.print(f"checkpoint: {ckpt_path}")

    return model
