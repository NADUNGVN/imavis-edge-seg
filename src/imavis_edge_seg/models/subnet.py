"""Static subnet extraction (RESEARCH_PLAN.md §5.3 "engine tĩnh thay vì dynamic graph").

`extract_subnet` copies the sliced weights for one elasticity level out of a trained
`PaceSegSupernet` into a plain `nn.Sequential`-style module built only from
`nn.Conv2d`/`nn.BatchNorm2d`/`nn.ReLU`/nearest-neighbor-free static upsample -- no
`SlimmableConv2d`, no per-call channel slicing, no level branching left. This is the
module that gets `torch.onnx.export`-ed (see `export.py`) and handed to
TensorRT/DLA/Hailo DFC; the supernet itself is only a training-time artifact.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from imavis_edge_seg.config import ElasticityLevel
from imavis_edge_seg.models.blocks import (
    ElasticDWSeparableBlock,
    ElasticPointwiseConvBNAct,
    ElasticStem,
    SlimmableConv2d,
)
from imavis_edge_seg.models.supernet import PaceSegSupernet


def _static_conv(conv: SlimmableConv2d, active_in: int, active_out: int) -> nn.Conv2d:
    with torch.no_grad():
        if conv.depthwise:
            weight = conv.weight[:active_out].clone()
            groups = active_out
        else:
            weight = conv.weight[:active_out, :active_in].clone()
            groups = 1
        static = nn.Conv2d(
            in_channels=active_in,
            out_channels=active_out,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=groups,
            bias=conv.bias is not None,
        )
        static.weight.copy_(weight)
        if conv.bias is not None:
            assert static.bias is not None
            static.bias.copy_(conv.bias[:active_out])
    return static


def _static_bn(bn_dict: nn.ModuleDict, level: ElasticityLevel) -> nn.BatchNorm2d:
    source = bn_dict[level]
    assert isinstance(source, nn.BatchNorm2d)
    static = nn.BatchNorm2d(source.num_features)
    static.load_state_dict(source.state_dict())
    static.eval()
    return static


class _StaticStemBlock(nn.Module):
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = self.act(self.bn(self.conv(x)))
        return result


class _StaticDWSeparableBlock(nn.Module):
    def __init__(
        self, depthwise: nn.Conv2d, bn1: nn.BatchNorm2d, pointwise: nn.Conv2d, bn2: nn.BatchNorm2d
    ) -> None:
        super().__init__()
        self.depthwise = depthwise
        self.bn1 = bn1
        self.pointwise = pointwise
        self.bn2 = bn2
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        result: Tensor = self.act(self.bn2(self.pointwise(x)))
        return result


def _dw_blocks(stage: nn.ModuleList, depth: int) -> list[ElasticDWSeparableBlock]:
    blocks: list[ElasticDWSeparableBlock] = []
    for module in list(stage)[:depth]:
        assert isinstance(module, ElasticDWSeparableBlock)
        blocks.append(module)
    return blocks


def _extract_stem(stem: ElasticStem, level: ElasticityLevel) -> _StaticStemBlock:
    c_out = stem.out_channels_per_level[level]
    conv = _static_conv(stem.conv, active_in=3, active_out=c_out)
    bn = _static_bn(stem.bn.bns, level)
    return _StaticStemBlock(conv, bn)


def _extract_dwseparable(
    block: ElasticDWSeparableBlock, level: ElasticityLevel
) -> _StaticDWSeparableBlock:
    c_in = block.in_channels_per_level[level]
    c_out = block.out_channels_per_level[level]
    depthwise = _static_conv(block.depthwise, active_in=c_in, active_out=c_in)
    bn1 = _static_bn(block.bn1.bns, level)
    pointwise = _static_conv(block.pointwise, active_in=c_in, active_out=c_out)
    bn2 = _static_bn(block.bn2.bns, level)
    return _StaticDWSeparableBlock(depthwise, bn1, pointwise, bn2)


def _extract_pointwise(
    block: ElasticPointwiseConvBNAct, level: ElasticityLevel
) -> _StaticStemBlock:
    c_in = block.in_channels_per_level[level]
    c_out = block.out_channels_per_level[level]
    conv = _static_conv(block.conv, active_in=c_in, active_out=c_out)
    bn = _static_bn(block.bn.bns, level)
    return _StaticStemBlock(conv, bn)


class StaticPaceSegSubnet(nn.Module):
    """Plain-op static segmentation network for one elasticity level. Safe to
    `torch.onnx.export` with a fixed input shape (see `export.py`)."""

    def __init__(self, supernet: PaceSegSupernet, level: ElasticityLevel) -> None:
        super().__init__()
        self.level = level
        depth = supernet.config.depth_blocks[level]

        self.stem = _extract_stem(supernet.stem, level)
        self.stage1 = nn.ModuleList(
            [_extract_dwseparable(b, level) for b in _dw_blocks(supernet.stage1, depth)]
        )
        self.stage2 = nn.ModuleList(
            [_extract_dwseparable(b, level) for b in _dw_blocks(supernet.stage2, depth)]
        )
        self.stage3 = nn.ModuleList(
            [_extract_dwseparable(b, level) for b in _dw_blocks(supernet.stage3, depth)]
        )

        self.reduce3 = _extract_pointwise(supernet.reduce3, level)
        self.fuse2 = _extract_dwseparable(supernet.fuse2, level)
        self.reduce2 = _extract_pointwise(supernet.reduce2, level)
        self.fuse1 = _extract_dwseparable(supernet.fuse1, level)
        self.reduce1 = _extract_pointwise(supernet.reduce1, level)
        self.fuse0 = _extract_dwseparable(supernet.fuse0, level)

        c_stem = supernet.stem_channels_per_level[level]
        self.classifier = _static_conv(
            supernet.classifier, active_in=c_stem, active_out=supernet.config.num_classes
        )

        self.eval()

    def _run_stage(self, stage: nn.ModuleList, x: Tensor) -> Tensor:
        for block in stage:
            x = block(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        s0 = self.stem(x)
        s1 = self._run_stage(self.stage1, s0)
        s2 = self._run_stage(self.stage2, s1)
        s3 = self._run_stage(self.stage3, s2)

        d2 = nn.functional.interpolate(self.reduce3(s3), scale_factor=2.0, mode="bilinear") + s2
        d2 = self.fuse2(d2)
        d1 = nn.functional.interpolate(self.reduce2(d2), scale_factor=2.0, mode="bilinear") + s1
        d1 = self.fuse1(d1)
        d0 = nn.functional.interpolate(self.reduce1(d1), scale_factor=2.0, mode="bilinear") + s0
        d0 = self.fuse0(d0)

        out = nn.functional.interpolate(d0, scale_factor=2.0, mode="bilinear")
        result: Tensor = self.classifier(out)
        return result


def extract_subnet(supernet: PaceSegSupernet, level: ElasticityLevel) -> StaticPaceSegSubnet:
    """Materialize a static, compiler-safe subnet for one elasticity level from a
    (trained or freshly-initialized) shared-weight supernet."""
    return StaticPaceSegSubnet(supernet, level)
