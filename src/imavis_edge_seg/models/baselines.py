"""Required-baseline model factory (`docs/RESEARCH_PLAN.md` §7): independently-trained
reference models the supernet's extracted subnets must be compared against. Each
entry returns a plain `nn.Module` whose `forward(image) -> logits` matches
`(B, num_classes, H, W)` at the *input* resolution, so it drops into the same
`training.baseline_trainer` loop and `training.losses.boundary_aware_segmentation_loss`
used for the supernet, with no supernet-specific machinery (no elasticity, no
sandwich rule, no distillation).

Only `mobilenetv3_deeplabv3` is implemented so far -- it needs no custom architecture
code (torchvision ships it), so it validates the whole baseline-training pipeline
first. The other six required baselines (Fast-SCNN, BiSeNetV2, PIDNet-S or
DDRNet-23-slim, SegFormer-B0, HARD, UCPNet) have no upstream off-the-shelf
implementation in this project's dependencies yet and are listed as explicit
`NotImplementedError`s rather than silently missing from `BASELINE_NAMES`, so a
caller iterating that list finds out immediately which ones still need work.
"""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

BASELINE_NAMES = (
    "mobilenetv3_deeplabv3",
    "fast_scnn",
    "bisenetv2",
    "pidnet_s",
    "segformer_b0",
    "hard",
    "ucpnet",
)


class _DeepLabV3MobileNetV3(nn.Module):
    """Thin wrapper: torchvision's `DeepLabV3` returns a dict (`{"out": ..., "aux":
    ...}`, aux only when `aux_loss=True`) already upsampled to the input's (H, W) --
    unwrap to a plain logits tensor so this matches every other model in this
    project's forward signature."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)

    def forward(self, image: Tensor) -> Tensor:
        out: Tensor = self.model(image)["out"]
        return out


def build_baseline_model(name: str, num_classes: int = 19) -> nn.Module:
    if name == "mobilenetv3_deeplabv3":
        return _DeepLabV3MobileNetV3(num_classes)
    if name in BASELINE_NAMES:
        raise NotImplementedError(
            f"baseline {name!r} is a required baseline (RESEARCH_PLAN.md §7) but has no "
            "implementation yet -- add it to models/baselines.py, it is not a config error"
        )
    raise ValueError(f"unknown baseline {name!r}, expected one of {BASELINE_NAMES}")
