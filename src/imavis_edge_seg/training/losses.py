"""Segmentation, boundary-aware, and in-place-distillation losses
(RESEARCH_PLAN.md §5.3 A). All functions take logits shaped (B, num_classes, H, W) and
integer trainId masks shaped (B, H, W); `IGNORE_INDEX` pixels are excluded everywhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from imavis_edge_seg.data.labels import IGNORE_INDEX


def resize_image(image: Tensor, size: tuple[int, int]) -> Tensor:
    if tuple(image.shape[-2:]) == tuple(size):
        return image
    return F.interpolate(image, size=size, mode="bilinear", align_corners=False)


def resize_mask(mask: Tensor, size: tuple[int, int]) -> Tensor:
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask
    resized = F.interpolate(mask.unsqueeze(1).float(), size=size, mode="nearest")
    return resized.squeeze(1).long()


def boundary_weight_map(target: Tensor) -> Tensor:
    """1.0 everywhere valid, 3x on label-boundary pixels (a pixel whose trainId differs
    from a 4-neighbor), 0 on ignored pixels -- protects small/thin classes
    (pedestrian, pole, sign, road boundary) at small elasticity levels."""
    padded = F.pad(target.unsqueeze(1).float(), (1, 1, 1, 1), mode="replicate").squeeze(1)
    up = padded[:, :-2, 1:-1]
    down = padded[:, 2:, 1:-1]
    left = padded[:, 1:-1, :-2]
    right = padded[:, 1:-1, 2:]
    center = target.float()
    is_boundary = (center != up) | (center != down) | (center != left) | (center != right)
    weight = torch.where(is_boundary, torch.full_like(center, 3.0), torch.ones_like(center))
    return torch.where(target == IGNORE_INDEX, torch.zeros_like(weight), weight)


def boundary_aware_segmentation_loss(
    logits: Tensor, target: Tensor, boundary_weight: float = 1.0
) -> Tensor:
    per_pixel = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX, reduction="none")
    valid = (target != IGNORE_INDEX).float()
    denom = valid.sum().clamp_min(1.0)
    base = (per_pixel * valid).sum() / denom

    weights = boundary_weight_map(target)
    weighted_denom = weights.sum().clamp_min(1.0)
    boundary_term = (per_pixel * weights).sum() / weighted_denom

    return base + boundary_weight * boundary_term


def distillation_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    valid_mask: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """KL(teacher || student) over valid (non-ignore) pixels. `teacher_logits` must
    already be `.detach()`-ed by the caller -- no gradient flows into the teacher path
    from this loss."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    kl_per_pixel = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
    valid = valid_mask.float()
    denom = valid.sum().clamp_min(1.0)
    return (kl_per_pixel * valid).sum() / denom * (temperature**2)
