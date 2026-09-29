"""Cheap per-image risk score (`docs/RESEARCH_PLAN.md` Contribution 3): the router
runs one cheap pass (typically the "tiny" elasticity level) and turns its own output
confidence into a single scalar risk estimate per image, rather than training a
separate probe network -- entropy of the softmax distribution is a standard,
well-understood uncertainty signal and needs no extra model or training data of its
own. This raw score is *uncalibrated*; `calibrator.py` turns it into an expected-error
estimate fit on real validation data, which is what the router policy actually acts on.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from imavis_edge_seg.data.labels import IGNORE_INDEX


def compute_risk_score(logits: Tensor, target: Tensor | None = None) -> Tensor:
    """`logits`: `(B, C, H, W)` raw (pre-softmax) scores. Returns `(B,)`: mean
    per-pixel softmax entropy (nats), one scalar per image in the batch. If `target`
    (`(B, H, W)` integer class ids) is given, only pixels where `target !=
    IGNORE_INDEX` count towards the mean. The target-masked branch is retained for
    diagnostics only; calibrator fitting must call `compute_deployment_risk_score`
    so its feature is identical to real inference, where ground truth is absent."""
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)  # (B, H, W)
    if target is not None:
        valid = (target != IGNORE_INDEX).float()
        denom = valid.sum(dim=(1, 2)).clamp_min(1)
        return (entropy * valid).sum(dim=(1, 2)) / denom
    return entropy.mean(dim=(1, 2))


def compute_deployment_risk_score(logits: Tensor) -> Tensor:
    """Return the entropy feature available to the deployed router.

    This is the canonical feature for both calibrator fitting and inference.  It
    deliberately accepts no target tensor, which makes accidental ground-truth
    masking impossible at the call site.  Calibration targets may still exclude
    ``IGNORE_INDEX`` pixels; supervised targets and deployable input features do
    not need to use the same pixel domain.
    """
    return compute_risk_score(logits)
