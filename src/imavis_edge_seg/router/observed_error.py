"""Per-image observed error -- the calibration *target* `calibrator.fit_risk_calibrator`
fits `risk_probe.compute_risk_score`'s raw entropy against. Uses per-image pixel
misclassification rate (1 - pixel accuracy), not mIoU: mIoU is only meaningful
accumulated over many images (a class absent from one image has undefined IoU there),
so it cannot serve as a *per-image* label the way this router calibration step needs.
"""

from __future__ import annotations

from torch import Tensor

from imavis_edge_seg.data.labels import IGNORE_INDEX


def compute_per_image_error(pred: Tensor, target: Tensor) -> Tensor:
    """`pred`/`target`: `(B, H, W)` integer class ids (`pred` from `logits.argmax(1)`).
    Returns `(B,)`: fraction of non-`IGNORE_INDEX` pixels where `pred != target`, one
    scalar per image -- masking matches `risk_probe.compute_risk_score`'s `target`-given
    branch, so a risk score and its observed error are computed over the same pixels."""
    valid = target != IGNORE_INDEX
    wrong = (pred != target) & valid
    denom = valid.sum(dim=(1, 2)).clamp_min(1)
    return wrong.sum(dim=(1, 2)).float() / denom.float()
