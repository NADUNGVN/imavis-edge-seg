"""Fits a monotonic mapping from `risk_probe.compute_risk_score`'s raw entropy score
to expected error, on validation data only -- `docs/RESEARCH_PLAN.md` §5's
"Calibration is fit on validation data only" applies here directly: fitting on test
data would leak information the router isn't allowed to have.

Equal-count (quantile) binning + a per-bin mean, like
`evaluation.calibration.CalibrationAccumulator`'s ECE bins, then a cumulative-max pass
to force monotonicity -- deliberately not a full isotonic-regression library call (no
new dependency), so is a simplification: a genuine risk signal should already be
monotonic in expectation, and the cumulative-max only needs to correct small-sample
noise, not fix a fundamentally broken signal. If bins are visibly non-monotonic before
correction, the raw risk score isn't a good risk signal and the fix here is a better
score, not a better calibrator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RiskCalibrator:
    bin_edges: np.ndarray  # (num_bins + 1,), quantile-spaced, first/last are +-inf
    bin_expected_error: np.ndarray  # (num_bins,), monotonic non-decreasing

    def predict(self, raw_score: float) -> float:
        bin_index = int(
            np.clip(np.searchsorted(self.bin_edges, raw_score, side="right") - 1, 0, len(self.bin_expected_error) - 1)
        )
        return float(self.bin_expected_error[bin_index])

    def to_dict(self) -> dict[str, Any]:
        """`+-inf` edges round-trip through JSON via Python's `float("inf")` support
        in `json.dumps`/`json.loads` (not strict JSON, but stdlib handles it)."""
        return {
            "bin_edges": self.bin_edges.tolist(),
            "bin_expected_error": self.bin_expected_error.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskCalibrator:
        return cls(
            bin_edges=np.asarray(data["bin_edges"], dtype=np.float64),
            bin_expected_error=np.asarray(data["bin_expected_error"], dtype=np.float64),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> RiskCalibrator:
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_risk_calibrator(
    raw_scores: np.ndarray | list[float],
    errors: np.ndarray | list[float],
    num_bins: int = 10,
) -> RiskCalibrator:
    """`raw_scores`/`errors` are paired per-image (raw risk, observed error -- e.g.
    `1 - mIoU` or a per-image misclassification rate) samples from a validation set,
    same length. Raises `ValueError` on empty input rather than returning a degenerate
    calibrator."""
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if len(raw_scores) == 0:
        raise ValueError("cannot fit a calibrator with zero validation samples")
    if len(raw_scores) != len(errors):
        raise ValueError(f"raw_scores ({len(raw_scores)}) and errors ({len(errors)}) length mismatch")

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(raw_scores, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    bin_expected_error = np.full(num_bins, np.nan)
    for i in range(num_bins):
        if i < num_bins - 1:
            mask = (raw_scores >= bin_edges[i]) & (raw_scores < bin_edges[i + 1])
        else:
            mask = (raw_scores >= bin_edges[i]) & (raw_scores <= bin_edges[i + 1])
        if mask.any():
            bin_expected_error[i] = errors[mask].mean()

    # Empty bins (possible with few validation samples / many bins): fill from the
    # nearest lower filled bin, or 0.0 if it's the very first bin and still empty.
    for i in range(num_bins):
        if np.isnan(bin_expected_error[i]):
            bin_expected_error[i] = bin_expected_error[i - 1] if i > 0 else 0.0

    bin_expected_error = np.maximum.accumulate(bin_expected_error)
    return RiskCalibrator(bin_edges=bin_edges, bin_expected_error=bin_expected_error)
