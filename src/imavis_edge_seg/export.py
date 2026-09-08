"""ONNX export for a static subnet -- the artifact that Week 1-2's compiler smoke test
(scripts/compiler_smoke_test.md) feeds into TensorRT / Xavier DLA / Hailo DFC.

Batch size is always 1 and the spatial shape is fixed at export time (RESEARCH_PLAN.md
§5.3 -- static engines, not a dynamic graph; DLA in particular does not support dynamic
shapes).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def export_subnet_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_height: int,
    input_width: int,
    opset: int = 17,
) -> Path:
    model.eval()
    dummy = torch.zeros(1, 3, input_height, input_width)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["image"],
        output_names=["seg_logits"],
        opset_version=opset,
        dynamic_axes=None,  # static shape by design -- see module docstring
        do_constant_folding=True,
        dynamo=False,  # TorchScript-based exporter -- avoids an onnxscript dependency
    )
    return output_path
