"""Quantization-aware training (QAT), Contribution 6 / `docs/RESEARCH_PLAN.md` §5.2,
§11 go bar ("INT8-QAT loses <=~1.0-1.5 mIoU vs FP32 per key subnet").

Scope of this first pass: per-tensor, *dynamic* (recomputed every forward from the
tensor's own observed min/max, not a running-average observer) symmetric fake
quantization applied to every `nn.Conv2d`'s weight and input activation, via
`torch.fake_quantize_per_tensor_affine` (a differentiable op with a built-in
straight-through gradient estimator -- the forward pass rounds to INT8 levels, the
backward pass passes gradients through as if no quantization happened, which is the
standard QAT training trick). This targets the *baseline* models
(`models/baseline_architectures.py`, `models/baselines.py`), which use plain
`nn.Conv2d` -- not yet the supernet's `SlimmableConv2d` (`models/blocks.py`), whose
per-level active-channel slicing needs its own quantization-range handling and is
follow-up work, not done here.

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


def apply_qat(model: nn.Module, num_bits: int = 8) -> nn.Module:
    """Recursively replaces every plain `nn.Conv2d` submodule of `model` with a
    `QATConv2d` sharing its weight/bias, in place, and returns `model`. Safe to call
    only once per model: `QATConv2d` **is** an `nn.Conv2d` (that is what preserves
    `state_dict` key names), so a second pass would re-wrap already-wrapped layers,
    stacking fake-quantization redundantly -- callers needing idempotence should
    check `isinstance(module, QATConv2d)` themselves before calling again."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d) and not isinstance(child, QATConv2d):
            setattr(model, name, QATConv2d(child, num_bits))
        else:
            apply_qat(child, num_bits)
    return model
