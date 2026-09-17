"""Supernet training loop orchestration: optimizer, LR schedule, the sandwich-rule
step from `training.step`, periodic logging and checkpointing. Kept separate from
`scripts/train_supernet.py` so the loop itself is importable/testable without going
through the CLI.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterator
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
from imavis_edge_seg.training.data import build_calibration_dataloader
from imavis_edge_seg.training.losses import resize_image
from imavis_edge_seg.training.quantization import apply_qat, run_calibration
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
    qat: bool = False,
    init_checkpoint: Path | None = None,
    calibrate: bool = False,
    calibration_images: int = 200,
) -> PaceSegSupernet:
    console = console or Console()
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)

    supernet = PaceSegSupernet(config.supernet)

    # Mirrors training.baseline_trainer.run_baseline_training's identical block:
    # load FP32 weights before apply_qat takes over those (now FP32-initialized)
    # Parameter objects, and .to(device) last so QAT-converted layers land on the
    # right device too.
    if init_checkpoint is not None:
        init_state = load_checkpoint(init_checkpoint, map_location="cpu")
        supernet.load_state_dict(init_state["model_state_dict"])
        console.print(f"initialized from {init_checkpoint} (step={init_state['step']})")
    if qat:
        supernet = apply_qat(supernet)  # type: ignore[assignment]
        console.print(
            "QAT enabled: all nn.Conv2d and SlimmableConv2d layers fake-quantized (INT8)"
        )
    supernet = supernet.to(device)

    # Calibrated (not dynamic) activation quantization ranges, RESEARCH_PLAN.md §5.2:
    # fit *before* the resume-checkpoint check below, on a real calibration set
    # spanning day/night/rain/fog/snow, from whatever weights are in `supernet` right
    # now (either freshly-initialized-from-`init_checkpoint`, or the FP32 supernet if
    # this is the very first QAT step from scratch). If a resumable checkpoint is
    # found next, its own state_dict (calibration buffers included) overwrites this --
    # correct, since resuming a run should continue with exactly the calibration it
    # already committed to, not restart calibration mid-fine-tune.
    if calibrate:
        if not qat:
            raise ValueError("--calibrate requires --qat (calibration only affects QAT quantization ranges)")
        calibration_loader = build_calibration_dataloader(config, max_images=calibration_images)
        levels = config.supernet.levels

        def _calibration_calls() -> Iterator[Callable[[], None]]:
            for image, _mask in calibration_loader:
                image = image.to(device)
                for level in levels:
                    height, width = config.supernet.input_resolutions[level]
                    resized = resize_image(image, (height, width))

                    def _call(x: torch.Tensor = resized, level: str = level) -> None:
                        supernet(x, level)

                    yield _call

        run_calibration(supernet, _calibration_calls())
        console.print(
            f"calibrated activation quantization ranges from {len(calibration_loader.dataset)} "  # type: ignore[arg-type]
            f"images x {len(levels)} levels"
        )

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
