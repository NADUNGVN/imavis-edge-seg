"""Runs a supernet at one elasticity level over a dataloader in eval mode and
accumulates mIoU. Deliberately evaluates the shared-weight supernet directly (not a
static `extract_subnet` copy) -- `test_models.py` already verifies the two produce
numerically identical output, so there is no accuracy difference, and avoiding the
extraction step keeps evaluation fast to call repeatedly during/after training.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from imavis_edge_seg.config import ElasticityLevel
from imavis_edge_seg.data.labels import NUM_CLASSES
from imavis_edge_seg.evaluation.metrics import ConfusionMatrixAccumulator, EvalResult


@torch.no_grad()
def evaluate_level(
    supernet: nn.Module,
    level: ElasticityLevel,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str = "cpu",
    num_classes: int = NUM_CLASSES,
) -> EvalResult:
    supernet.eval()
    accumulator = ConfusionMatrixAccumulator(num_classes=num_classes)
    for image, mask in dataloader:
        image = image.to(device)
        mask = mask.to(device)
        logits = supernet(image, level)
        pred = logits.argmax(dim=1)
        accumulator.update(pred, mask)
    return accumulator.compute()
