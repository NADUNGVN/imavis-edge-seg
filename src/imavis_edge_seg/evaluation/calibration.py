"""Calibration/reliability metrics (`docs/RESEARCH_PLAN.md` §8: "ECE, NLL/Brier,
risk-coverage curve + AURC ... selective mIoU at fixed coverage/risk targets") --
the prerequisite this project's calibrated visual-risk router (Contribution 3) needs
a real confidence signal to calibrate against, before any router logic can be built.

Streaming/binned, like `metrics.ConfusionMatrixAccumulator`: NLL and Brier are exact
running sums (no approximation), but the risk-coverage curve and ECE bin per-pixel
confidence into `num_bins` fixed-width buckets rather than keeping every pixel's raw
confidence in memory -- a full Cityscapes+ACDC eval pass is tens to hundreds of
millions of pixels, too much to hold as a flat list. This makes the risk-coverage
curve/AURC an approximation at bin resolution, standard practice for this metric at
this scale, not full pixel-level precision.

UIoU (uncertainty-aware IoU, using ACDC's own uncertain-region annotations) is not
implemented here -- it needs those annotations wired into the ACDC dataset loader
first, which hasn't happened yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from imavis_edge_seg.data.labels import IGNORE_INDEX, NUM_CLASSES


@dataclass
class CalibrationResult:
    ece: float
    nll: float
    brier: float
    aurc: float
    bin_confidence: list[float]  # mean predicted confidence per bin, low -> high
    bin_accuracy: list[float]  # empirical accuracy per bin (NaN if the bin is empty)
    bin_count: list[int]
    num_pixels: int


class CalibrationAccumulator:
    """Accumulates confidence/correctness statistics across many batches of a
    segmentation eval pass, then computes ECE/NLL/Brier/AURC once at the end -- same
    accumulate-then-compute discipline as `ConfusionMatrixAccumulator`, for the same
    reason (a metric averaged per-batch is not the same number as one computed over
    the whole accumulated distribution, and is not the correct way to report this)."""

    def __init__(self, num_classes: int = NUM_CLASSES, num_bins: int = 15) -> None:
        self.num_classes = num_classes
        self.num_bins = num_bins
        self._bin_count = torch.zeros(num_bins, dtype=torch.int64)
        self._bin_confidence_sum = torch.zeros(num_bins, dtype=torch.float64)
        self._bin_correct_sum = torch.zeros(num_bins, dtype=torch.float64)
        self._nll_sum = 0.0
        self._brier_sum = 0.0
        self._pixel_count = 0

    def update(self, logits: Tensor, target: Tensor) -> None:
        """`logits`: (B, num_classes, H, W) raw (pre-softmax) scores. `target`: (B, H,
        W) integer class ids; pixels equal to `IGNORE_INDEX` are excluded, same
        convention as `metrics.compute_confusion_matrix`."""
        valid = target != IGNORE_INDEX
        if not bool(valid.any()):
            return

        probs = F.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)
        confidence = confidence[valid]
        pred = pred[valid]
        target_valid = target[valid]
        correct = (pred == target_valid).double()

        bin_edges = torch.linspace(0.0, 1.0, self.num_bins + 1, device=confidence.device)
        # Rightmost bin is [1-1/num_bins, 1] inclusive of confidence==1.0.
        bin_index = torch.clamp(
            torch.bucketize(confidence, bin_edges[1:-1], right=False), 0, self.num_bins - 1
        )
        self._bin_count += torch.bincount(bin_index, minlength=self.num_bins).cpu()
        self._bin_confidence_sum += (
            torch.bincount(bin_index, weights=confidence.double(), minlength=self.num_bins).cpu()
        )
        self._bin_correct_sum += (
            torch.bincount(bin_index, weights=correct, minlength=self.num_bins).cpu()
        )

        probs_valid = _gather_valid_probs(probs, valid)
        log_probs_valid = torch.log(probs_valid.clamp_min(1e-12))
        nll_per_pixel = -log_probs_valid.gather(1, target_valid.unsqueeze(1)).squeeze(1)
        self._nll_sum += float(nll_per_pixel.sum())

        one_hot = F.one_hot(target_valid, num_classes=self.num_classes).double()
        brier_per_pixel = ((probs_valid.double() - one_hot) ** 2).sum(dim=1)
        self._brier_sum += float(brier_per_pixel.sum())

        self._pixel_count += int(valid.sum())

    def compute(self) -> CalibrationResult:
        n = max(self._pixel_count, 1)
        bin_confidence: list[float] = []
        bin_accuracy: list[float] = []
        bin_count: list[int] = []
        ece = 0.0
        for i in range(self.num_bins):
            count = int(self._bin_count[i])
            bin_count.append(count)
            if count == 0:
                bin_confidence.append(float("nan"))
                bin_accuracy.append(float("nan"))
                continue
            mean_conf = float(self._bin_confidence_sum[i] / count)
            mean_acc = float(self._bin_correct_sum[i] / count)
            bin_confidence.append(mean_conf)
            bin_accuracy.append(mean_acc)
            ece += (count / n) * abs(mean_conf - mean_acc)

        aurc = _aurc_from_bins(self._bin_count, self._bin_correct_sum, self._bin_confidence_sum)

        return CalibrationResult(
            ece=ece,
            nll=self._nll_sum / n,
            brier=self._brier_sum / n,
            aurc=aurc,
            bin_confidence=bin_confidence,
            bin_accuracy=bin_accuracy,
            bin_count=bin_count,
            num_pixels=self._pixel_count,
        )


def _gather_valid_probs(probs: Tensor, valid: Tensor) -> Tensor:
    """`probs`: (B, C, H, W). `valid`: (B, H, W) bool. Returns (N, C) -- one row per
    valid pixel, channel dim last, ready for `.gather(1, target.unsqueeze(1))`."""
    channels = probs.shape[1]
    probs_bhwc = probs.permute(0, 2, 3, 1).reshape(-1, channels)
    valid_flat = valid.reshape(-1)
    return probs_bhwc[valid_flat]


def _aurc_from_bins(
    bin_count: Tensor, bin_correct_sum: Tensor, bin_confidence_sum: Tensor
) -> float:
    """Area under the risk-coverage curve, approximated at bin resolution: process
    bins from *highest* confidence to lowest (a selective-prediction system would
    reject the least-confident pixels first), accumulating coverage (fraction of
    pixels kept) and risk (error rate among kept pixels) bin by bin. Risk is only
    defined once at least one pixel is covered -- there is no real "coverage=0" risk
    value to anchor a trapezoid at -- so this is a right-Riemann/step-function sum
    (each bin's cumulative risk applies to the coverage interval it just closed), the
    standard discretization for AURC (Geifman & El-Yaniv selective-classification
    formulation), not a trapezoid against a fictitious (0, 0) point."""
    total = int(bin_count.sum())
    if total == 0:
        return float("nan")

    order = torch.argsort(bin_confidence_sum / bin_count.clamp_min(1), descending=True)
    cumulative_count = 0
    cumulative_correct = 0.0
    prev_coverage = 0.0
    aurc = 0.0
    for i in order.tolist():
        count = int(bin_count[i])
        if count == 0:
            continue
        cumulative_count += count
        cumulative_correct += float(bin_correct_sum[i])
        coverage = cumulative_count / total
        risk = 1.0 - cumulative_correct / cumulative_count
        aurc += (coverage - prev_coverage) * risk
        prev_coverage = coverage
    return aurc
