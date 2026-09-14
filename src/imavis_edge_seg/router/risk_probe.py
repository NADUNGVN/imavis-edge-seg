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
    IGNORE_INDEX` count towards the mean -- pass it when *fitting* a calibrator (so
    the risk score is computed over exactly the pixels the observed error will also
    be computed over); omit it at real inference time, when there is no ground truth
    to know which pixels would have been ignored."""
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)  # (B, H, W)
    if target is not None:
        valid = (target != IGNORE_INDEX).float()
        denom = valid.sum(dim=(1, 2)).clamp_min(1)
        return (entropy * valid).sum(dim=(1, 2)) / denom
    return entropy.mean(dim=(1, 2))
