"""Confusion-matrix-based mIoU, the vision-quality metric `RESEARCH_PLAN.md` §8 asks
for (overall + per-class + per-condition mIoU). Accumulates over an entire dataset
before computing IoU, rather than averaging per-batch IoU, which is the standard
(and correct) way to handle classes that are rare or absent in some batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from imavis_edge_seg.data.labels import IGNORE_INDEX, NUM_CLASSES, TRAIN_ID_TO_NAME


def compute_confusion_matrix(pred: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """`pred`/`target` are integer class-id tensors of the same shape (any number of
    dims -- typically (B, H, W)). Pixels where `target == IGNORE_INDEX` are excluded.
    Returns an int64 (num_classes, num_classes) matrix, rows=ground truth, cols=pred."""
    valid = target != IGNORE_INDEX
    pred = pred[valid]
    target = target[valid]
    indices = target.long() * num_classes + pred.long()
    matrix = torch.bincount(indices, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def iou_per_class(confusion: Tensor) -> Tensor:
    """NaN for a class with zero union (absent from both prediction and ground truth
    in the accumulated data) -- callers should `nanmean` rather than `mean`."""
    true_positive = confusion.diagonal().float()
    predicted = confusion.sum(dim=0).float()
    actual = confusion.sum(dim=1).float()
    union = predicted + actual - true_positive
    return torch.where(union > 0, true_positive / union, torch.full_like(union, float("nan")))


@dataclass
class EvalResult:
    miou: float
    per_class_iou: dict[str, float]
    num_pixels: int


class ConfusionMatrixAccumulator:
    """Accumulates a confusion matrix across many batches (an entire eval split),
    then computes mIoU once at the end -- do not average per-batch mIoU values."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self.matrix: Tensor = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        self._pixel_count = 0

    def update(self, pred: Tensor, target: Tensor) -> None:
        self.matrix += compute_confusion_matrix(pred, target, self.num_classes).to(
            self.matrix.device
        )
        self._pixel_count += int((target != IGNORE_INDEX).sum())

    def compute(self) -> EvalResult:
        per_class = iou_per_class(self.matrix)
        miou = float(torch.nanmean(per_class))
        names = {
            TRAIN_ID_TO_NAME.get(i, str(i)): float(per_class[i]) for i in range(self.num_classes)
        }
        return EvalResult(miou=miou, per_class_iou=names, num_pixels=self._pixel_count)
