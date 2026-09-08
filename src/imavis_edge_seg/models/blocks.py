"""Compiler-safe building blocks for the elastic supernet.

Every op used here is on the "candidate intersection" list from
`docs/RESEARCH_PLAN.md` §5.1 / `scripts/compiler_smoke_test.md`: plain and depthwise
Conv2d, BatchNorm2d, ReLU, static-factor bilinear resize, elementwise add. Nothing here
is dynamic at inference time -- `extract_subnet` (subnet.py) turns a trained supernet
into a plain `nn.Conv2d`/`nn.BatchNorm2d` graph with no slicing left, which is what
actually gets exported to ONNX/TensorRT/HEF.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from imavis_edge_seg.config import ElasticityLevel


class SlimmableConv2d(nn.Module):
    """A Conv2d whose weight is stored at the widest ("large") channel count and
    sliced down to `active_in`/`active_out` channels per forward call. `depthwise=True`
    stores a per-channel [C, 1, k, k] filter bank (groups == active channel count at
    forward time), matching a standard depthwise-separable block's first stage.
    """

    def __init__(
        self,
        max_in_channels: int,
        max_out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        depthwise: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if depthwise and max_in_channels != max_out_channels:
            raise ValueError("depthwise SlimmableConv2d requires max_in == max_out")
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.depthwise = depthwise
        weight_in = 1 if depthwise else max_in_channels
        self.weight = nn.Parameter(torch.empty(max_out_channels, weight_in, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(max_out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor, active_in: int, active_out: int) -> Tensor:
        if self.depthwise:
            weight = self.weight[:active_out]
            groups = active_out
        else:
            weight = self.weight[:active_out, :active_in]
            groups = 1
        bias = self.bias[:active_out] if self.bias is not None else None
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding, groups=groups)


class SlimmableBatchNorm2d(nn.Module):
    """Switchable batch norm: one independent BatchNorm2d per elasticity level, since
    activation statistics differ meaningfully by width (standard practice for
    slimmable/elastic networks -- sharing BN affine params across widths destabilizes
    training)."""

    def __init__(self, channels_per_level: dict[ElasticityLevel, int]) -> None:
        super().__init__()
        self.bns = nn.ModuleDict(
            {level: nn.BatchNorm2d(channels) for level, channels in channels_per_level.items()}
        )

    def forward(self, x: Tensor, level: ElasticityLevel) -> Tensor:
        bn = self.bns[level]
        assert isinstance(bn, nn.BatchNorm2d)
        return cast(Tensor, bn(x))


class ElasticStem(nn.Module):
    """3x3 stride-2 standard conv from fixed 3-channel RGB input to the stem's
    per-level channel count."""

    def __init__(self, out_channels_per_level: dict[ElasticityLevel, int]) -> None:
        super().__init__()
        max_out = max(out_channels_per_level.values())
        self.conv = SlimmableConv2d(3, max_out, kernel_size=3, stride=2, padding=1)
        self.bn = SlimmableBatchNorm2d(out_channels_per_level)
        self.act = nn.ReLU(inplace=True)
        self.out_channels_per_level = out_channels_per_level

    def forward(self, x: Tensor, level: ElasticityLevel) -> Tensor:
        c_out = self.out_channels_per_level[level]
        x = self.conv(x, active_in=3, active_out=c_out)
        x = self.bn(x, level)
        return cast(Tensor, self.act(x))


class ElasticDWSeparableBlock(nn.Module):
    """Depthwise-separable block: depthwise 3x3 (given stride) then pointwise 1x1,
    each per-level channel count driven by the precomputed channel plan."""

    def __init__(
        self,
        in_channels_per_level: dict[ElasticityLevel, int],
        out_channels_per_level: dict[ElasticityLevel, int],
        stride: int,
    ) -> None:
        super().__init__()
        self.in_channels_per_level = in_channels_per_level
        self.out_channels_per_level = out_channels_per_level
        max_in = max(in_channels_per_level.values())
        max_out = max(out_channels_per_level.values())
        self.depthwise = SlimmableConv2d(
            max_in, max_in, kernel_size=3, stride=stride, padding=1, depthwise=True
        )
        self.bn1 = SlimmableBatchNorm2d(in_channels_per_level)
        self.pointwise = SlimmableConv2d(max_in, max_out, kernel_size=1, stride=1, padding=0)
        self.bn2 = SlimmableBatchNorm2d(out_channels_per_level)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, level: ElasticityLevel) -> Tensor:
        c_in = self.in_channels_per_level[level]
        c_out = self.out_channels_per_level[level]
        x = self.depthwise(x, active_in=c_in, active_out=c_in)
        x = self.bn1(x, level)
        x = self.act(x)
        x = self.pointwise(x, active_in=c_in, active_out=c_out)
        x = self.bn2(x, level)
        return cast(Tensor, self.act(x))


class ElasticPointwiseConvBNAct(nn.Module):
    """1x1 conv + BN + act, used for decoder channel-reduction ("lateral") steps."""

    def __init__(
        self,
        in_channels_per_level: dict[ElasticityLevel, int],
        out_channels_per_level: dict[ElasticityLevel, int],
    ) -> None:
        super().__init__()
        self.in_channels_per_level = in_channels_per_level
        self.out_channels_per_level = out_channels_per_level
        max_in = max(in_channels_per_level.values())
        max_out = max(out_channels_per_level.values())
        self.conv = SlimmableConv2d(max_in, max_out, kernel_size=1, stride=1, padding=0)
        self.bn = SlimmableBatchNorm2d(out_channels_per_level)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, level: ElasticityLevel) -> Tensor:
        c_in = self.in_channels_per_level[level]
        c_out = self.out_channels_per_level[level]
        x = self.conv(x, active_in=c_in, active_out=c_out)
        x = self.bn(x, level)
        return cast(Tensor, self.act(x))


def static_upsample_2x(x: Tensor) -> Tensor:
    """Fixed-factor bilinear upsample -- compiler-safe static resize, not a dynamic
    size lookup (see compiler_smoke_test.md)."""
    return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
