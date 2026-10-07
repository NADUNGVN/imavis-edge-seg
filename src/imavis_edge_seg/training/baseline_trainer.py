"""Training loop for a single fixed-architecture required baseline (`RESEARCH_PLAN.md`
§7) -- plain segmentation + boundary-aware loss at the supernet's largest configured
resolution, no elasticity sampling, no in-place distillation. Deliberately mirrors
`training.trainer.run_training`'s structure (same optimizer/schedule/checkpoint
conventions) so results are comparable and reviewers see one consistent training
recipe, not two different ones invented for baselines vs. the proposed method.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from rich.console import Console
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from imavis_edge_seg.config import ElasticityLevel, ExperimentConfig
from imavis_edge_seg.models.baselines import build_baseline_model
from imavis_edge_seg.training.speed import SpeedSettings
from imavis_edge_seg.training.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from imavis_edge_seg.training.data import build_calibration_dataloader
from imavis_edge_seg.training.losses import boundary_aware_segmentation_loss
from imavis_edge_seg.training.quantization import CalibrationObserver, apply_qat, run_calibration
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
    qat: bool = False,
    init_checkpoint: Path | None = None,
    model: nn.Module | None = None,
    calibrate: bool = False,
    calibration_images: int = 200,
    calibration_observer: CalibrationObserver = "max",
    calibration_percentile: float = 0.999,
    calibration_momentum: float = 0.9,
    calibration_level: ElasticityLevel | None = None,
) -> nn.Module:
    console = console or Console()
    torch.manual_seed(config.seed)

    # `model` lets a caller (e.g. an exported-subnet QAT fine-tuning script) hand in an
    # already-built model -- e.g. models.subnet.extract_subnet's output -- instead of
    # building `baseline_name` fresh via build_baseline_model. Both take a single `x`
    # (no elasticity `level` argument), so the rest of this loop is unchanged either way.
    if model is None:
        model = build_baseline_model(baseline_name, num_classes=config.supernet.num_classes)

    # RESEARCH_PLAN.md §5.2's "FP32 teacher -> shared supernet -> QAT INT8" workflow:
    # start QAT from an already-trained FP32 checkpoint's weights, not from scratch.
    # Order matters -- load the FP32 weights into the *plain* model first, then
    # apply_qat takes over those same (now FP32-initialized) weight/bias Parameter
    # objects, rather than the freshly-initialized random ones; `.to(device)` last so
    # every parameter (including apply_qat's newly-constructed QATConv2d instances)
    # ends up on the right device, not just the ones that existed before conversion.
    if init_checkpoint is not None:
        init_state = load_checkpoint(init_checkpoint, map_location="cpu")
        model.load_state_dict(init_state["model_state_dict"])
        console.print(f"initialized from {init_checkpoint} (step={init_state['step']})")
    if qat:
        model = apply_qat(model)
        console.print(f"[{baseline_name}] QAT enabled: all nn.Conv2d layers fake-quantized (INT8)")
    model = model.to(device)

    # Calibrated (not dynamic) activation quantization ranges -- mirrors
    # training.trainer.run_training's identical block, minus the per-level loop (a
    # baseline/exported-subnet model has exactly one resolution/forward signature).
    if calibrate:
        if not qat:
            raise ValueError("--calibrate requires --qat (calibration only affects QAT quantization ranges)")
        calibration_loader = build_calibration_dataloader(
            config, max_images=calibration_images, level=calibration_level
        )

        def _calibration_calls() -> Iterator[Callable[[], None]]:
            for image, _mask in calibration_loader:
                image = image.to(device)

                def _call(x: torch.Tensor = image) -> None:
                    model(x)

                yield _call

        run_calibration(
            model,
            _calibration_calls(),
            observer=calibration_observer,
            percentile=calibration_percentile,
            momentum=calibration_momentum,
        )
        console.print(
            f"[{baseline_name}] calibrated activation quantization ranges from "
            f"{len(calibration_loader.dataset)} images (observer={calibration_observer})"  # type: ignore[arg-type]
        )

    optimizer = AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    speed = SpeedSettings(config, device, qat)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, config))

    # Resume if a checkpoint already exists for this experiment_id -- see
    # training.trainer.run_training's identical block for why.
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
            model.load_state_dict(checkpoint["model_state_dict"])
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
        return model

    # Train mode explicitly: models.subnet.StaticPaceSegSubnet (the pace_large baseline)
    # calls self.eval() in __init__, which otherwise freezes BatchNorm at its init
    # running stats for the whole run (2026-10-07 bug: PACE-Large ~0.33 mIoU).
    model.train()
    batches = _infinite_batches(dataloader)
    running_loss = 0.0

    for step in range(start_step + 1, config.training.max_steps + 1):
        image, mask = next(batches)
        image = image.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        with speed.autocast():
            logits = model(image)
            loss = boundary_aware_segmentation_loss(logits, mask, boundary_weight=config.training.boundary_loss_weight)
        speed.backward_and_step(loss, model, optimizer, config.training.grad_clip_norm)
        scheduler.step()

        running_loss += loss.detach()

        if step % config.training.log_interval_steps == 0:
            avg_loss = float(running_loss) / config.training.log_interval_steps
            running_loss = 0.0
            lr = scheduler.get_last_lr()[0]
            console.print(
                f"[{baseline_name}] step {step}/{config.training.max_steps} loss={avg_loss:.4f} lr={lr:.2e} "
                f"{speed.throughput(step, config.training.max_steps)}"
            )

        if step % config.training.checkpoint_interval_steps == 0 or step == config.training.max_steps:
            ckpt_path = output_dir / "checkpoints" / f"step_{step:08d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, config)
            console.print(f"checkpoint: {ckpt_path}")

    return model
