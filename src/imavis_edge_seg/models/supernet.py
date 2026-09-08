"""The elastic segmentation supernet (RESEARCH_PLAN.md contribution 1).

Encoder: stem + 3 downsampling stages of depthwise-separable blocks (Fast-SCNN-style).
Decoder: top-down FPN-style fusion back to input resolution.
Elastic axes: width (per-level channel plan, `channels.py`) and depth (per-level block
count within each stage, `supernet_config.depth_blocks`). Resolution elasticity is
handled by what the caller feeds in -- the network is fully convolutional.

`forward(x, level)` runs the shared-weight supernet at one elasticity level, used during
sandwich-rule training (sample multiple levels per step, backprop through all of them).
It is not itself compiler-safe (the per-call channel slicing is a Python-level branch,
not a static ONNX-exportable op) -- `subnet.extract_subnet` materializes a static,
single-level model for that.
"""

from __future__ import annotations

from torch import Tensor, nn

from imavis_edge_seg.config import ElasticityLevel, SupernetConfig
from imavis_edge_seg.models.blocks import (
    ElasticDWSeparableBlock,
    ElasticPointwiseConvBNAct,
    ElasticStem,
    SlimmableConv2d,
    static_upsample_2x,
)
from imavis_edge_seg.models.channels import ChannelPlan, build_channel_plan


def _make_stage(
    in_channels_per_level: dict[ElasticityLevel, int],
    out_channels_per_level: dict[ElasticityLevel, int],
    max_depth: int,
) -> nn.ModuleList:
    """First block downsamples (stride 2) and changes channel count in->out; remaining
    blocks (up to `max_depth`) refine at stride 1, out->out. A subnet at a shallower
    depth simply uses a prefix of this list (see PaceSegSupernet._run_stage)."""
    blocks = [ElasticDWSeparableBlock(in_channels_per_level, out_channels_per_level, stride=2)]
    for _ in range(max_depth - 1):
        blocks.append(
            ElasticDWSeparableBlock(out_channels_per_level, out_channels_per_level, stride=1)
        )
    return nn.ModuleList(blocks)


class PaceSegSupernet(nn.Module):
    def __init__(self, supernet_config: SupernetConfig) -> None:
        super().__init__()
        self.config = supernet_config
        self.channel_plan: ChannelPlan = build_channel_plan(supernet_config)
        max_depth = max(supernet_config.depth_blocks.values())

        self.stem = ElasticStem(self.channel_plan.stem)
        self.stage1 = _make_stage(self.channel_plan.stem, self.channel_plan.stage1, max_depth)
        self.stage2 = _make_stage(self.channel_plan.stage1, self.channel_plan.stage2, max_depth)
        self.stage3 = _make_stage(self.channel_plan.stage2, self.channel_plan.stage3, max_depth)

        self.reduce3 = ElasticPointwiseConvBNAct(self.channel_plan.stage3, self.channel_plan.stage2)
        self.fuse2 = ElasticDWSeparableBlock(self.channel_plan.stage2, self.channel_plan.stage2, stride=1)
        self.reduce2 = ElasticPointwiseConvBNAct(self.channel_plan.stage2, self.channel_plan.stage1)
        self.fuse1 = ElasticDWSeparableBlock(self.channel_plan.stage1, self.channel_plan.stage1, stride=1)
        self.reduce1 = ElasticPointwiseConvBNAct(self.channel_plan.stage1, self.channel_plan.stem)
        self.fuse0 = ElasticDWSeparableBlock(self.channel_plan.stem, self.channel_plan.stem, stride=1)

        max_stem = self.channel_plan.max_channels("stem")
        self.classifier = SlimmableConv2d(
            max_stem, supernet_config.num_classes, kernel_size=1, bias=True
        )
        self.stem_channels_per_level = self.channel_plan.stem

    def _run_stage(self, stage: nn.ModuleList, x: Tensor, level: ElasticityLevel) -> Tensor:
        depth = self.config.depth_blocks[level]
        for block in list(stage)[:depth]:
            x = block(x, level)
        return x

    def forward(self, x: Tensor, level: ElasticityLevel) -> Tensor:
        s0 = self.stem(x, level)
        s1 = self._run_stage(self.stage1, s0, level)
        s2 = self._run_stage(self.stage2, s1, level)
        s3 = self._run_stage(self.stage3, s2, level)

        d2 = static_upsample_2x(self.reduce3(s3, level)) + s2
        d2 = self.fuse2(d2, level)
        d1 = static_upsample_2x(self.reduce2(d2, level)) + s1
        d1 = self.fuse1(d1, level)
        d0 = static_upsample_2x(self.reduce1(d1, level)) + s0
        d0 = self.fuse0(d0, level)

        out = static_upsample_2x(d0)
        c_in = self.stem_channels_per_level[level]
        result: Tensor = self.classifier(out, active_in=c_in, active_out=self.config.num_classes)
        return result
