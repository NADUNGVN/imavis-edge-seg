"""Quantization-aware training (QAT), Contribution 6 / `docs/RESEARCH_PLAN.md` §5.2,
§11 go bar ("INT8-QAT loses <=~1.0-1.5 mIoU vs FP32 per key subnet").

Per-tensor symmetric fake quantization applied to every `nn.Conv2d`'s (and the
supernet's `SlimmableConv2d`'s) weight and input activation, via
`torch.fake_quantize_per_tensor_affine` (a differentiable op with a built-in
straight-through gradient estimator -- the forward pass rounds to INT8 levels, the
backward pass passes gradients through as if no quantization happened, which is the
standard QAT training trick).

Two activation-quantization-range modes, selectable per layer:

- **Dynamic** (the original, still the default): the scale is recomputed from the
  input tensor's own observed min/max on *every* forward call. A real simplification
  vs. production QAT -- documented as such since this module's introduction -- because
  it reacts to whatever one batch happens to look like rather than a range fit from a
  broad, representative sample.
- **Calibrated** (added 2026-09-17, `run_calibration`): a fixed scale, observed once
  from a forward pass over a real calibration set spanning day/night/rain/fog/snow
  (`RESEARCH_PLAN.md` §5.2's explicit requirement), then frozen for every subsequent
  forward call. This is what closes the gap the dynamic mode's docstring always
  flagged as open. Weight quantization is *not* given a calibrated mode -- a weight
  tensor's own min/max is always exactly known (no data-dependent uncertainty the way
  an activation's range has), so computing it dynamically per call is already exact,
  not a simplification.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

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
    max_abs = x.detach().abs().max()
    return fake_quantize_tensor_with_max(x, max_abs, num_bits)


def fake_quantize_tensor_with_max(x: Tensor, max_abs: Tensor | float, num_bits: int = 8) -> Tensor:
    """Same fake quantization as `fake_quantize_tensor`, but the scale is derived
    from a *given* `max_abs` (e.g. one observed and frozen during calibration)
    rather than `x`'s own value this call -- what a calibrated (not dynamic)
    activation range needs."""
    quant_min = -(2 ** (num_bits - 1))
    quant_max = 2 ** (num_bits - 1) - 1
    max_abs_value = float(max_abs)
    if max_abs_value == 0.0:
        return x
    scale = max_abs_value / quant_max
    zero_point = 0
    return torch.fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max)


def _init_calibration_buffers(module: nn.Module) -> None:
    """Registers the frozen max-abs-activation buffer *and* the `calibrated` flag
    itself as `state_dict`-persisted buffers -- both must survive a checkpoint
    save/reload, or `evaluate_supernet.py --qat` loading a *calibrated* checkpoint
    would silently fall back to dynamic quantization (the max would reload
    correctly, but a plain-Python `calibrated` flag would not, since state_dict
    only carries tensor/buffer state) -- exactly the kind of silent train/eval
    mismatch this project has caught before and taken care not to repeat.
    `calibrating` (only ever true *during* a `run_calibration` call, never at
    training-resume or eval time) is a plain, non-persistent flag -- there is
    nothing meaningful to save mid-calibration."""
    module.register_buffer("calibrated_max", torch.tensor(0.0))
    module.register_buffer("calibrated", torch.tensor(False))


def _quantize_activation(module: QATConv2d | QATSlimmableConv2d, x: Tensor, num_bits: int) -> Tensor:
    """Shared input-activation quantization dispatch for QATConv2d/
    QATSlimmableConv2d.forward, reflecting `module`'s calibration state: while
    `calibrating`, only observe (grow `calibrated_max`, don't quantize yet); once
    `calibrated`, quantize with that frozen max; otherwise (the original, default
    behavior), quantize dynamically from this call's own input."""
    if module.calibrating:
        with torch.no_grad():
            batch_max = x.detach().abs().max()
            if batch_max > module.calibrated_max:
                module.calibrated_max.fill_(float(batch_max))
        return x
    if bool(module.calibrated):
        return fake_quantize_tensor_with_max(x, module.calibrated_max, num_bits)
    return fake_quantize_tensor(x, num_bits)


class QATConv2d(nn.Conv2d):
    """A drop-in replacement for an existing `nn.Conv2d`, fake-quantizing its weight
    and input on every forward call. Deliberately *subclasses* `nn.Conv2d` (rather
    than wrapping one as a submodule) and takes over the original's `weight`/`bias`
    `Parameter` objects directly -- so a converted model's `state_dict` keys are
    identical to the original's (`<name>.weight`, `<name>.bias`, no added nesting).
    That is what makes the RESEARCH_PLAN.md §5.2 workflow ("FP32 teacher -> shared
    supernet -> QAT INT8") possible: an FP32 checkpoint can be loaded into a model
    both before and after `apply_qat`, since the parameter names never change."""

    calibrating: bool
    calibrated: Tensor
    calibrated_max: Tensor

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
        _init_calibration_buffers(self)
        self.calibrating = False

    def forward(self, x: Tensor) -> Tensor:
        q_input = _quantize_activation(self, x, self.num_bits)
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

    calibrating: bool
    calibrated: Tensor
    calibrated_max: Tensor

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
        _init_calibration_buffers(self)
        self.calibrating = False

    def forward(self, x: Tensor, active_in: int, active_out: int) -> Tensor:
        if self.depthwise:
            weight = self.weight[:active_out]
            groups = active_out
        else:
            weight = self.weight[:active_out, :active_in]
            groups = 1
        bias = self.bias[:active_out] if self.bias is not None else None
        q_input = _quantize_activation(self, x, self.num_bits)
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


def run_calibration(model: nn.Module, forward_calls: Iterable[Callable[[], None]]) -> None:
    """Fits a *calibrated* (not dynamic) input-activation quantization range for
    every `QATConv2d`/`QATSlimmableConv2d` in `model`: each `forward_calls` entry is
    a zero-arg callable that should trigger exactly one forward pass through
    `model` (e.g. `lambda: model(x)`, or `lambda: supernet(x, level)` -- callers
    typically pass one such call per (calibration image, elasticity level) pair, so
    every level's own active-channel slice gets calibrated too). Every layer's
    observed max activation magnitude grows monotonically across all calls, then is
    frozen once every call has run -- callers should pass a real, representative
    calibration set (`RESEARCH_PLAN.md` §5.2: day/night/rain/fog/snow), never the
    same data a policy or checkpoint is later evaluated against.

    Raises `ValueError` if `model` has no QAT-converted layers -- calling this
    before `apply_qat` would otherwise silently do nothing."""
    modules: list[QATConv2d | QATSlimmableConv2d] = [
        m for m in model.modules() if isinstance(m, QATConv2d | QATSlimmableConv2d)
    ]
    if not modules:
        raise ValueError("model has no QATConv2d/QATSlimmableConv2d layers to calibrate -- call apply_qat first")
    for m in modules:
        m.calibrating = True
        m.calibrated.fill_(False)
        m.calibrated_max.fill_(0.0)  # discard any previous calibration, start fresh
    with torch.no_grad():
        for call in forward_calls:
            call()
    for m in modules:
        m.calibrating = False
        m.calibrated.fill_(True)
