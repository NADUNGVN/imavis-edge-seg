"""From-scratch implementations of required baselines (`docs/RESEARCH_PLAN.md` §7) that
have no off-the-shelf checkpoint in this project's dependencies. Plain `nn.Conv2d`/
`nn.BatchNorm2d` throughout -- no elasticity, no slimmable machinery (that's specific to
`models/blocks.py`'s supernet); each of these is one fixed static architecture, trained
and evaluated exactly like the supernet's "large" level but never resized at inference.

Random-init (no pretrained backbone) for both -- the point of comparison against
`mobilenetv3_deeplabv3` (which does use an ImageNet-pretrained backbone, see
`baselines.py`'s docstring) is to also have at least one required baseline trained under
the exact same from-scratch condition as the supernet, so the "gap <=2 mIoU"
(RESEARCH_PLAN.md §11) comparison isn't confounded by initialization for every baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _conv_bn_act(
    in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1
) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _dw_separable(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        _conv_bn_act(in_channels, in_channels, 3, stride, groups=in_channels),
        _conv_bn_act(in_channels, out_channels, 1, 1),
    )


class _InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual bottleneck, the Global Feature Extractor's
    building block in Fast-SCNN."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expansion: int) -> None:
        super().__init__()
        hidden = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            _conv_bn_act(in_channels, hidden, 1),
            _conv_bn_act(hidden, hidden, 3, stride, groups=hidden),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class _PyramidPooling(nn.Module):
    """Pyramid Pooling Module (bin sizes 1/2/3/6), Fast-SCNN's global context stage."""

    def __init__(self, channels: int, bins: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        branch_channels = channels // len(bins)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for bin_size in bins
            ]
        )
        self.project = _conv_bn_act(channels + branch_channels * len(bins), channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        pooled = [x]
        for stage in self.stages:
            y = stage(x)
            pooled.append(F.interpolate(y, size=size, mode="bilinear", align_corners=False))
        return self.project(torch.cat(pooled, dim=1))  # type: ignore[no-any-return]


class FastSCNN(nn.Module):
    """Fast-SCNN (Poudel et al. 2019): Learning-to-Downsample -> Global Feature
    Extractor (inverted-residual stages + pyramid pooling) -> Feature Fusion ->
    classifier. Channel widths match the paper's default config."""

    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()
        # Learning to Downsample: 3 stride-2 stages, 8x total downsample.
        self.downsample = nn.Sequential(
            _conv_bn_act(3, 32, 3, stride=2),
            _dw_separable(32, 48, stride=2),
            _dw_separable(48, 64, stride=2),
        )

        # Global Feature Extractor: 3 inverted-residual stages + pyramid pooling.
        def _stage(in_channels: int, out_channels: int, stride: int, n: int) -> nn.Sequential:
            layers = [_InvertedResidual(in_channels, out_channels, stride, expansion=6)]
            layers += [_InvertedResidual(out_channels, out_channels, 1, expansion=6) for _ in range(n - 1)]
            return nn.Sequential(*layers)

        self.gfe = nn.Sequential(
            _stage(64, 64, stride=2, n=3),
            _stage(64, 96, stride=2, n=3),
            _stage(96, 128, stride=1, n=3),
            _PyramidPooling(128),
        )

        # Feature Fusion Module: high-res (downsample output, 64ch) + low-res (gfe
        # output, 128ch, needs 4x upsample to match downsample's spatial size) -> add.
        self.ffm_low_res_project = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.ffm_high_res_project = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.ffm_act = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
            _dw_separable(128, 128),
            _dw_separable(128, 128),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, image: Tensor) -> Tensor:
        input_size = image.shape[-2:]
        high_res = self.downsample(image)
        low_res = self.gfe(high_res)
        low_res_up = F.interpolate(low_res, size=high_res.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.ffm_act(self.ffm_high_res_project(high_res) + self.ffm_low_res_project(low_res_up))
        logits = self.classifier(fused)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class _GatherExpansionLayer(nn.Module):
    """BiSeNetV2's Semantic Branch building block: 3x3 conv, then a depthwise 3x3
    "expansion" (channel count x6, stride 1 or 2), then a 1x1 "project" back down,
    plus a residual/shortcut path. Simplified vs. the paper's exact stride-2 shortcut
    (here: avgpool+1x1 instead of a second depthwise conv) -- structurally the same
    gather-expand-project shape, not a claim of exact op-for-op fidelity."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 6) -> None:
        super().__init__()
        hidden = in_channels * expansion
        self.conv = _conv_bn_act(in_channels, in_channels, 3, 1)
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(hidden),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride == 1 and in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(3, stride, 1) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.project(self.expand(self.conv(x)))
        return self.act(out + self.shortcut(x))  # type: ignore[no-any-return]


class _DetailBranch(nn.Module):
    """3 stride-2 stages, plain convs -- kept at 1/8 input resolution throughout,
    preserves spatial detail the low-channel Semantic Branch discards."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(_conv_bn_act(3, 64, 3, 2), _conv_bn_act(64, 64, 3, 1))
        self.stage2 = nn.Sequential(
            _conv_bn_act(64, 64, 3, 2), _conv_bn_act(64, 64, 3, 1), _conv_bn_act(64, 64, 3, 1)
        )
        self.stage3 = nn.Sequential(
            _conv_bn_act(64, 128, 3, 2), _conv_bn_act(128, 128, 3, 1), _conv_bn_act(128, 128, 3, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stage3(self.stage2(self.stage1(x)))  # type: ignore[no-any-return]


class _SemanticBranch(nn.Module):
    """Stem block (two parallel downsample paths, concat) -> 3 Gather-and-Expansion
    stages -> Context Embedding Block (global-context add). Channel width follows the
    paper's ~1/4 ratio vs. the Detail Branch."""

    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = _conv_bn_act(3, 16, 3, 2)
        self.stem_branch_conv = nn.Sequential(_conv_bn_act(16, 8, 1, 1), _conv_bn_act(8, 16, 3, 2))
        self.stem_branch_pool = nn.MaxPool2d(3, 2, 1)
        self.stem_project = _conv_bn_act(32, 16, 3, 1)

        self.stage3 = nn.Sequential(
            _GatherExpansionLayer(16, 32, stride=2), _GatherExpansionLayer(32, 32, stride=1)
        )
        self.stage4 = nn.Sequential(
            _GatherExpansionLayer(32, 64, stride=2), _GatherExpansionLayer(64, 64, stride=1)
        )
        self.stage5 = nn.Sequential(
            _GatherExpansionLayer(64, 128, stride=2),
            _GatherExpansionLayer(128, 128, stride=1),
            _GatherExpansionLayer(128, 128, stride=1),
            _GatherExpansionLayer(128, 128, stride=1),
        )
        self.context_project = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128))

    def forward(self, x: Tensor) -> Tensor:
        stem_out = self.stem_conv(x)
        stem = self.stem_project(
            torch.cat([self.stem_branch_conv(stem_out), self.stem_branch_pool(stem_out)], dim=1)
        )
        feat = self.stage5(self.stage4(self.stage3(stem)))
        context = feat.mean(dim=(2, 3), keepdim=True)
        return feat + self.context_project(context)  # type: ignore[no-any-return]


class _BilateralGuidedAggregation(nn.Module):
    """Fuses Detail Branch (high-res, spatial detail) and Semantic Branch (low-res,
    context) outputs -- simplified to project-both-to-detail's-resolution-then-add
    rather than the paper's full mutual-attention gating, but keeps the bilateral
    (both branches contribute, at the detail branch's resolution) structure."""

    def __init__(self, detail_channels: int = 128, semantic_channels: int = 128, out_channels: int = 128) -> None:
        super().__init__()
        self.detail_project = nn.Sequential(
            nn.Conv2d(detail_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)
        )
        self.semantic_project = nn.Sequential(
            nn.Conv2d(semantic_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, detail: Tensor, semantic: Tensor) -> Tensor:
        semantic_up = F.interpolate(semantic, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        return self.act(self.detail_project(detail) + self.semantic_project(semantic_up))  # type: ignore[no-any-return]


class BiSeNetV2(nn.Module):
    """BiSeNet V2 (Yu et al. 2021): Detail Branch + Semantic Branch, fused by a
    (simplified) Bilateral Guided Aggregation layer, then a segmentation head. Only the
    main head is used here -- the paper's auxiliary "booster" heads (training-only,
    dropped at inference) aren't wired into this project's single-loss training loop."""

    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()
        self.detail = _DetailBranch()
        self.semantic = _SemanticBranch()
        self.aggregation = _BilateralGuidedAggregation()
        self.head = nn.Sequential(_conv_bn_act(128, 128, 3, 1), nn.Conv2d(128, num_classes, 1))

    def forward(self, image: Tensor) -> Tensor:
        input_size = image.shape[-2:]
        detail_feat = self.detail(image)
        semantic_feat = self.semantic(image)
        fused = self.aggregation(detail_feat, semantic_feat)
        logits = self.head(fused)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


class _BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (2x conv3x3 + residual) -- DDRNet's trunk/branch
    building block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))  # type: ignore[no-any-return]


class _Bottleneck(nn.Module):
    """ResNet Bottleneck (1x1 reduce -> 3x3 -> 1x1 expand x2), used for DDRNet's final
    stage on both branches."""

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = mid_channels * 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + self.shortcut(x))  # type: ignore[no-any-return]


class _DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module: like a PSPNet-style pyramid pooling
    module, but each pooled scale is fused (added, after upsampling) into the *next*
    finer scale before its own 3x3 conv, cascading global context down to the finest
    branch instead of pooling every branch independently in parallel."""

    def __init__(self, in_channels: int, branch_channels: int, out_channels: int) -> None:
        super().__init__()
        self.scale0 = nn.Sequential(nn.Conv2d(in_channels, branch_channels, 1, bias=False), nn.BatchNorm2d(branch_channels))
        pool_specs = [(5, 2, 2), (9, 4, 4), (17, 8, 8)]  # (kernel, stride, padding)
        self.pools = nn.ModuleList(
            [nn.AvgPool2d(k, s, p) for k, s, p in pool_specs] + [nn.AdaptiveAvgPool2d(1)]
        )
        self.pool_projects = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(in_channels, branch_channels, 1, bias=False), nn.BatchNorm2d(branch_channels))
                for _ in range(4)
            ]
        )
        self.process = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                )
                for _ in range(4)
            ]
        )
        self.compression = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        branches = [self.scale0(x)]
        prev = branches[0]
        for pool, project, process in zip(self.pools, self.pool_projects, self.process, strict=True):
            pooled = F.interpolate(project(pool(x)), size=size, mode="bilinear", align_corners=False)
            prev = process(pooled + prev)
            branches.append(prev)
        out: Tensor = self.compression(torch.cat(branches, dim=1))
        return out + self.shortcut(x)  # type: ignore[no-any-return]


class DDRNetSlim(nn.Module):
    """DDRNet-23-slim (Hong et al. 2021, "Deep Dual-Resolution Networks for Real-time
    and Accurate Semantic Segmentation of Road Scenes"): a shared ResNet-style stem/
    trunk splits into a low-resolution branch (keeps downsampling for global context)
    and a high-resolution branch (stays at 1/8 input resolution throughout), with two
    bilateral fusion points between them, a DAPPM head on the low-res branch's output,
    and a final add + segmentation head. "-slim" channel widths (32 base / 64 high-res
    branch), matching the paper's lightweight variant."""

    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_act(3, 32, 3, 2), _conv_bn_act(32, 32, 3, 2)
        )
        self.layer1 = nn.Sequential(_BasicBlock(32, 32), _BasicBlock(32, 32))
        self.layer2 = nn.Sequential(_BasicBlock(32, 64, stride=2), _BasicBlock(64, 64))

        self.layer3_low = nn.Sequential(_BasicBlock(64, 128, stride=2), _BasicBlock(128, 128))
        self.layer4_low = nn.Sequential(_BasicBlock(128, 256, stride=2), _BasicBlock(256, 256))
        self.layer5_low = _Bottleneck(256, 256, stride=2)  # -> 512 channels

        self.layer3_high = nn.Sequential(_BasicBlock(64, 64), _BasicBlock(64, 64))
        self.layer4_high = nn.Sequential(_BasicBlock(64, 64), _BasicBlock(64, 64))
        self.layer5_high = _Bottleneck(64, 64, stride=1)  # -> 128 channels

        # Bilateral fusion 1 (after layer3_*): low(128,1/16) <-> high(64,1/8)
        self.compress1 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64))
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128))
        # Bilateral fusion 2 (after layer4_*): low(256,1/32) <-> high(64,1/8) -- high is
        # 4x coarser-resolution than low here (1/8 vs 1/32), so down2 needs two stride-2
        # steps, not one (unlike down1, where high/low were only 2x apart).
        self.compress2 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.dappm = _DAPPM(512, 96, 128)
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Sequential(_conv_bn_act(128, 128, 3, 1), nn.Conv2d(128, num_classes, 1))

    def forward(self, image: Tensor) -> Tensor:
        input_size = image.shape[-2:]
        trunk = self.layer2(self.layer1(self.stem(image)))  # 1/8, 64ch

        low = self.layer3_low(trunk)  # 1/16, 128ch
        high = self.layer3_high(trunk)  # 1/8, 64ch
        low, high = (
            self.act(low + self.down1(high)),
            self.act(high + F.interpolate(self.compress1(low), size=high.shape[-2:], mode="bilinear", align_corners=False)),
        )

        low = self.layer4_low(low)  # 1/32, 256ch
        high = self.layer4_high(high)  # 1/8, 64ch
        low, high = (
            self.act(low + self.down2(high)),
            self.act(high + F.interpolate(self.compress2(low), size=high.shape[-2:], mode="bilinear", align_corners=False)),
        )

        low = self.layer5_low(low)  # 1/64, 512ch
        high = self.layer5_high(high)  # 1/8, 128ch

        low_context = self.dappm(low)
        low_context_up = F.interpolate(low_context, size=high.shape[-2:], mode="bilinear", align_corners=False)
        fused = high + low_context_up
        logits = self.head(fused)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
