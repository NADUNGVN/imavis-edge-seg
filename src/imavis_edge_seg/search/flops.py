"""FLOPs-aware selection baseline (`docs/RESEARCH_PLAN.md` §7's "FLOPs-aware vs
latency-aware... selection" baseline axis, and RQ1's actual comparison point).
`search.pareto` selects by measured latency; this module counts FLOPs instead, so
RQ1's "hardware-aware selection beats FLOPs-aware selection" claim can be tested
quantitatively rather than only illustrated qualitatively (`reports/pareto_search_v1_20260912.md`).

Counts multiply-accumulate FLOPs (reported as 2x MACs, the common convention) for
every `nn.Conv2d`/`SlimmableConv2d` layer actually exercised in one forward call, via
forward hooks -- this naturally handles the supernet's per-level active-channel
slicing without extra bookkeeping, since a hook sees whatever input/output shape was
actually used in that specific call (no need to separately track which channels were
"active"). Other op types in the compiler-safe set (BatchNorm, ReLU, pooling, static
resize, add/concat) are FLOP-negligible next to convolution and are not counted --
a standard simplification in FLOPs-counting literature, not an oversight.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from imavis_edge_seg.config import ElasticityLevel
from imavis_edge_seg.models.blocks import SlimmableConv2d


def _flops_for_call(module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> int:
    if isinstance(module, SlimmableConv2d):
        in_channels = int(inputs[0].shape[1])
        out_channels = int(output.shape[1])
        kh = kw = module.kernel_size
        groups = out_channels if module.depthwise else 1
    elif isinstance(module, nn.Conv2d):
        in_channels = module.in_channels
        out_channels = module.out_channels
        kh, kw = module.kernel_size
        groups = module.groups
    else:
        return 0
    out_h, out_w = int(output.shape[-2]), int(output.shape[-1])
    macs = (in_channels // groups) * out_channels * kh * kw * out_h * out_w
    return int(2 * macs)


@torch.no_grad()
def count_flops(model: nn.Module, forward_fn: Callable[[], Tensor]) -> int:
    """Total FLOPs across every `nn.Conv2d`/`SlimmableConv2d` exercised while
    `forward_fn()` runs (`forward_fn` should call `model` exactly once, e.g.
    `lambda: model(x)` or `lambda: supernet(x, level)`)."""
    accum = 0

    def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Tensor) -> None:
        nonlocal accum
        accum += _flops_for_call(module, inputs, output)

    hooks = [
        m.register_forward_hook(_hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, SlimmableConv2d))
    ]
    try:
        forward_fn()
    finally:
        for h in hooks:
            h.remove()
    return accum


@dataclass(frozen=True)
class FlopsPoint:
    level: ElasticityLevel
    flops: int
    miou: float
    dataset: str


def build_flops_points(
    flops_by_level: dict[ElasticityLevel, int],
    miou_by_level: dict[str, dict[str, float]],
    dataset: str,
) -> list[FlopsPoint]:
    """Mirrors `search.pareto.build_pareto_points`'s join, but on FLOPs instead of
    measured latency. `miou_by_level` is `evaluate_supernet.py --output-json`'s
    `{level: {dataset: miou}}` shape."""
    points = []
    for level, flops in flops_by_level.items():
        miou = miou_by_level.get(level, {}).get(dataset)
        if miou is None:
            continue
        points.append(FlopsPoint(level=level, flops=flops, miou=miou, dataset=dataset))
    return points


def select_under_flops_budget(points: list[FlopsPoint], budget_flops: float) -> FlopsPoint | None:
    """Highest-mIoU point with `flops <= budget_flops` (ties broken by lower FLOPs);
    `None` if nothing fits the budget."""
    candidates = [p for p in points if p.flops <= budget_flops]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.miou, -p.flops))


def fit_flops_to_latency_rate(
    flops_by_level: dict[ElasticityLevel, int], latency_by_level: dict[ElasticityLevel, float]
) -> float:
    """Least-squares fit of `latency = k * flops` (through the origin) across every
    level both dicts share -- `k` (ms per FLOP) is what a FLOPs-only selection method
    calibrated on one device would have to carry over, unchanged, to predict cost on
    any other device (`scripts/flops_vs_latency_baseline.py`, RQ1's comparison
    point). Returns 0.0 if there is no overlapping level or all shared FLOPs are 0."""
    levels = [level for level in flops_by_level if level in latency_by_level]
    numerator = sum(flops_by_level[level] * latency_by_level[level] for level in levels)
    denominator = sum(flops_by_level[level] ** 2 for level in levels)
    return numerator / denominator if denominator > 0 else 0.0
