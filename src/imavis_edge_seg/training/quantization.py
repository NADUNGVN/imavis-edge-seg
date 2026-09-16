"""Quantization-aware training (QAT), Contribution 6 / `docs/RESEARCH_PLAN.md` §5.2,
§11 go bar ("INT8-QAT loses <=~1.0-1.5 mIoU vs FP32 per key subnet").

Scope of this first pass: per-tensor, *dynamic* (recomputed every forward from the
tensor's own observed min/max, not a running-average observer) symmetric fake
quantization applied to every `nn.Conv2d`'s weight and input activation, via
`torch.fake_quantize_per_tensor_affine` (a differentiable op with a built-in
straight-through gradient estimator -- the forward pass rounds to INT8 levels, the
backward pass passes gradients through as if no quantization happened, which is the
standard QAT training trick). Originally targeted only the *baseline* models
(`models/baseline_architectures.py`, plain `nn.Conv2d`); `QATSlimmableConv2d` (added
2026-09-16) extends the same mechanism to the supernet's `SlimmableConv2d`
(`models/blocks.py`), whose weight is sliced to the active level's channel count on
every forward call -- the fake-quantization scale is computed from that *sliced*
sub-tensor each call, so each elasticity level naturally gets its own dynamic range
without any extra bookkeeping.

A "dynamic per-tensor" range is a real simplification vs. production QAT (which
usually calibrates per-channel ranges from a running average over many batches) --
documented here, not hidden, because it changes how any INT8 accuracy number from
this module should be read: as an upper bound on how good a properly-calibrated QAT
setup could do, not the final number for a paper claim.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from imavis_edge_seg.models.blocks import SlimmableConv2d


def fake_quantize_tensor(x: Tensor, num_bits: int = 8) -> Tensor:
    """Symmetric per-tensor fake quantization: observes `x`'s own min/max this call,
    computes an INT`num_bits` scale from it, and rounds `x` to that grid via
    `torch.fake_quantize_per_tensor_affine` (STE gradient -- `d(fake_quantize(x))/dx`
    is treated as 1 within the representable range). A constant-zero tensor (e.g. an
    all-zero bias-free start) returns `x` unchanged rather than dividing by zero.
    """
    quant_min = -(2 ** (num_bits - 1))
    quant_max = 2 ** (num_bits - 1) - 1
    max_abs = x.detach().abs().max()
    if float(max_abs) == 0.0:
        return x
    scale = float(max_abs) / quant_max
    zero_point = 0
    return torch.fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max)


class QATConv2d(nn.Conv2d):
    """A drop-in replacement for an existing `nn.Conv2d`, fake-quantizing its weight
    and input on every forward call. Deliberately *subclasses* `nn.Conv2d` (rather
    than wrapping one as a submodule) and takes over the original's `weight`/`bias`
    `Parameter` objects directly -- so a converted model's `state_dict` keys are
    identical to the original's (`<name>.weight`, `<name>.bias`, no added nesting).
    That is what makes the RESEARCH_PLAN.md §5.2 workflow ("FP32 teacher -> shared
    supernet -> QAT INT8") possible: an FP32 checkpoint can be loaded into a model
    both before and after `apply_qat`, since the parameter names never change."""

    def __init__(self, conv: nn.Conv2d, num_bits: int = 8) -> None:
        super().__init__(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,  # type: ignore[arg-type]
            stride=conv.stride,  # type: ignore[arg-type]
            padding=conv.padding,  # type: ignore[arg-type]
            dilation=conv.dilation,  # type: ignore[arg-type]
            groups=conv.groups,
            bias=conv.bias is not None,
        )
        self.weight = conv.weight  # share the same Parameter, not a copy
        if conv.bias is not None:
            self.bias = conv.bias
        self.num_bits = num_bits

    def forward(self, x: Tensor) -> Tensor:
        q_input = fake_quantize_tensor(x, self.num_bits)
        q_weight = fake_quantize_tensor(self.weight, self.num_bits)
        return F.conv2d(
            q_input,
            q_weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QATSlimmableConv2d(SlimmableConv2d):
    """A drop-in replacement for an existing `SlimmableConv2d`, fake-quantizing its
    (per-call, per-level-sliced) weight and input on every forward call. Subclasses
    `SlimmableConv2d` directly (mirroring `QATConv2d`'s approach for plain
    `nn.Conv2d`) and takes over the original's `weight`/`bias` `Parameter` objects,
    so a converted supernet's `state_dict` keys are unchanged."""

    def __init__(self, conv: SlimmableConv2d, num_bits: int = 8) -> None:
        super().__init__(
            conv.max_in_channels,
            conv.max_out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            depthwise=conv.depthwise,
            bias=conv.bias is not None,
        )
        self.weight = conv.weight  # share the same Parameter, not a copy
        if conv.bias is not None:
            self.bias = conv.bias
        self.num_bits = num_bits

    def forward(self, x: Tensor, active_in: int, active_out: int) -> Tensor:
        if self.depthwise:
            weight = self.weight[:active_out]
            groups = active_out
        else:
            weight = self.weight[:active_out, :active_in]
            groups = 1
        bias = self.bias[:active_out] if self.bias is not None else None
        q_input = fake_quantize_tensor(x, self.num_bits)
        q_weight = fake_quantize_tensor(weight, self.num_bits)
        return F.conv2d(q_input, q_weight, bias, stride=self.stride, padding=self.padding, groups=groups)


def apply_qat(model: nn.Module, num_bits: int = 8) -> nn.Module:
    """Recursively replaces every plain `nn.Conv2d` submodule with a `QATConv2d`,
    and every `SlimmableConv2d` submodule with a `QATSlimmableConv2d`, each sharing
    the original's weight/bias, in place, and returns `model`. Safe to call only
    once per model: both QAT classes **are** subclasses of the type they replace
    (that is what preserves `state_dict` key names), so a second pass would re-wrap
    already-wrapped layers, stacking fake-quantization redundantly -- callers
    needing idempotence should check `isinstance(module, (QATConv2d,
    QATSlimmableConv2d))` themselves before calling again."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, QATConv2d):
            setattr(model, name, QATConv2d(child, num_bits))
        elif isinstance(child, SlimmableConv2d) and not isinstance(child, QATSlimmableConv2d):
            setattr(model, name, QATSlimmableConv2d(child, num_bits))
        else:
            apply_qat(child, num_bits)
    return model
