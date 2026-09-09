"""One sandwich-rule training step: sample levels, run the teacher (largest) first,
then each other sampled level with segmentation + boundary-aware + in-place-distillation
loss, sum, and return for a single `backward()` -- one optimizer step per batch, not one
per level, matching RESEARCH_PLAN.md §5.3 A/D.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from torch import Tensor

from imavis_edge_seg.config import ElasticityLevel, ExperimentConfig
from imavis_edge_seg.data.labels import IGNORE_INDEX
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.training.losses import (
    boundary_aware_segmentation_loss,
    distillation_kl_loss,
    resize_image,
    resize_mask,
)
from imavis_edge_seg.training.sandwich import sample_training_levels


@dataclass
class StepResult:
    total_loss: Tensor
    seg_losses: dict[ElasticityLevel, float] = field(default_factory=dict)
    distill_losses: dict[ElasticityLevel, float] = field(default_factory=dict)
    levels_trained: list[ElasticityLevel] = field(default_factory=list)


def train_step(
    supernet: PaceSegSupernet,
    image: Tensor,
    mask: Tensor,
    config: ExperimentConfig,
    rng: random.Random,
) -> StepResult:
    """`image`/`mask` are batches at the *largest* elasticity level's resolution
    (`config.supernet.input_resolutions[levels[-1]]`) -- every other sampled level is
    produced by downsampling here, not by re-reading the dataset at a different size."""
    levels = sample_training_levels(config.supernet, config.training, rng)
    teacher_level = config.supernet.levels[-1]

    total_loss: Tensor | None = None
    teacher_logits: Tensor | None = None
    result = StepResult(total_loss=image.new_zeros(()), levels_trained=levels)

    for level in levels:
        target_size = config.supernet.input_resolutions[level]
        level_image = resize_image(image, target_size)
        level_mask = resize_mask(mask, target_size)

        logits = supernet(level_image, level)
        seg_loss = boundary_aware_segmentation_loss(
            logits, level_mask, boundary_weight=config.training.boundary_loss_weight
        )
        result.seg_losses[level] = float(seg_loss.detach())
        level_loss = seg_loss

        if level == teacher_level:
            teacher_logits = logits.detach()
        elif teacher_logits is not None:
            teacher_resized = resize_image(teacher_logits, target_size)
            valid = level_mask != IGNORE_INDEX
            kd_loss = distillation_kl_loss(
                logits,
                teacher_resized,
                valid,
                temperature=config.training.distillation_temperature,
            )
            result.distill_losses[level] = float(kd_loss.detach())
            level_loss = level_loss + config.search.alpha_distill * kd_loss

        total_loss = level_loss if total_loss is None else total_loss + level_loss

    assert total_loss is not None  # `levels` is never empty
    result.total_loss = total_loss
    return result
