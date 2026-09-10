"""LR warmup + decay schedule, shared by the supernet trainer and the baseline
trainer (`RESEARCH_PLAN.md` §7 requires baselines trained under a comparable budget,
so both use the same warmup/decay shape, not independently-tuned schedules)."""

from __future__ import annotations

import math

from imavis_edge_seg.config import ExperimentConfig


def lr_lambda(step: int, config: ExperimentConfig) -> float:
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
