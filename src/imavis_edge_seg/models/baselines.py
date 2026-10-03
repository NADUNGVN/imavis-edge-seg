"""Required-baseline model factory (`docs/RESEARCH_PLAN.md` §7): independently-trained
reference models the supernet's extracted subnets must be compared against. Each
entry returns a plain `nn.Module` whose `forward(image) -> logits` matches
`(B, num_classes, H, W)` at the *input* resolution, so it drops into the same
`training.baseline_trainer` loop and `training.losses.boundary_aware_segmentation_loss`
used for the supernet, with no supernet-specific machinery (no elasticity, no
sandwich rule, no distillation).

`mobilenetv3_deeplabv3` needs no custom architecture code (torchvision ships it, with
an ImageNet-pretrained backbone -- see its wrapper's docstring for why that asymmetry
is deliberate and disclosed). `fast_scnn`, `bisenetv2`, `ddrnet23_slim` and
`segformer_b0` are from-scratch implementations in `baseline_architectures.py`
(random-init, matching the supernet's own from-scratch training condition).
RESEARCH_PLAN.md §7 asks for "PIDNet-S **or** DDRNet-23-slim" -- `ddrnet23_slim`
satisfies that; `pidnet_s` stays unimplemented (kept as a name in case the researcher
wants that specific one instead later, not because both are required). `hard` and
`ucpnet` are explicitly conditional in the plan ("if code/checkpoint reproducible" /
"if released in time") -- a web check on 2026-09-11 found no public code/checkpoint
release for either, so they stay `NotImplementedError` rather than a guessed
reconstruction of an unpublished architecture (which would risk misrepresenting
someone else's paper). Every unimplemented name raises `NotImplementedError`
explicitly rather than being silently missing from `BASELINE_NAMES`, so a caller
iterating that list finds out immediately which ones still need work (or, for
hard/ucpnet, why they're intentionally skipped).
"""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from imavis_edge_seg.models.baseline_architectures import (
    BiSeNetV2,
    DDRNetSlim,
    FastSCNN,
    SegformerB0,
)

BASELINE_NAMES = (
    "mobilenetv3_deeplabv3",
    "fast_scnn",
    "bisenetv2",
    "ddrnet23_slim",
    "pidnet_s",
    "segformer_b0",
    "hard",
    "ucpnet",
    "pace_large",
)


class _DeepLabV3MobileNetV3(nn.Module):
    """Thin wrapper: torchvision's `DeepLabV3` returns a dict (`{"out": ..., "aux":
    ...}`, aux only when `aux_loss=True`) already upsampled to the input's (H, W) --
    unwrap to a plain logits tensor so this matches every other model in this
    project's forward signature.

    Deliberate asymmetry, decided 2026-09-11: `weights=None` (segmentation head is
    random-init) but `weights_backbone` is left at torchvision's own default
    (ImageNet-pretrained MobileNetV3-Large) -- PACE-Seg's own supernet has no
    equivalent public pretrained checkpoint (bespoke architecture) and trains fully
    from scratch, so this baseline gets a real initialization advantage the proposed
    method does not. This is standard practice for how such baselines are normally
    deployed/reported, not an oversight -- but it must be disclosed as a methods-
    section caveat, and it makes the RESEARCH_PLAN.md §11 "subnet gap <=2 mIoU" go/no-go
    bar *harder* to clear (a pretrained baseline is a stronger opponent), not easier.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)

    def forward(self, image: Tensor) -> Tensor:
        out: Tensor = self.model(image)["out"]
        return out


def build_baseline_model(name: str, num_classes: int = 19) -> nn.Module:
    if name == "mobilenetv3_deeplabv3":
        return _DeepLabV3MobileNetV3(num_classes)
    if name == "fast_scnn":
        return FastSCNN(num_classes)
    if name == "bisenetv2":
        return BiSeNetV2(num_classes)
    if name == "ddrnet23_slim":
        return DDRNetSlim(num_classes)
    if name == "segformer_b0":
        return SegformerB0(num_classes)
    if name == "pace_large":
        # Standalone PACE-Seg large architecture (2026-10-03, RQ2 confound): the exact
        # static graph `extract_subnet(supernet, "large")` exports, built from a freshly
        # RANDOM-initialized supernet and then trained alone with the baseline recipe --
        # no weight sharing, sandwich sampling or distillation. Comparing it with the
        # large level of the shared elastic supernet isolates the shared-training effect.
        from imavis_edge_seg.config import SupernetConfig
        from imavis_edge_seg.models.subnet import extract_subnet
        from imavis_edge_seg.models.supernet import PaceSegSupernet

        return extract_subnet(PaceSegSupernet(SupernetConfig(num_classes=num_classes)), "large")
    if name in BASELINE_NAMES:
        raise NotImplementedError(
            f"baseline {name!r} is a required baseline (RESEARCH_PLAN.md §7) but has no "
            "implementation yet -- add it to models/baselines.py, it is not a config error"
        )
    raise ValueError(f"unknown baseline {name!r}, expected one of {BASELINE_NAMES}")
