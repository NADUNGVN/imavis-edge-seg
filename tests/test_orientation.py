"""Regression tests for the 2026-10-03 input-orientation bug: every elasticity level
must feed the network a landscape (H < W) tensor, matching Cityscapes/ACDC frames."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from imavis_edge_seg.config import SupernetConfig, load_config
from imavis_edge_seg.data.transforms import SegmentationResizeToTensor


def test_default_resolutions_are_landscape() -> None:
    config = load_config("configs/experiment/default.yaml")
    for level, (height, width) in config.supernet.input_resolutions.items():
        assert width == 2 * height, f"{level}: expected 2:1 landscape, got H={height} W={width}"
    assert config.supernet.input_resolutions["tiny"] == (192, 384)
    assert config.supernet.input_resolutions["large"] == (512, 1024)


def test_portrait_resolution_is_rejected() -> None:
    with pytest.raises(ValueError, match="portrait"):
        SupernetConfig(input_resolutions={"tiny": (384, 192)})


def test_eval_transform_produces_landscape_tensor() -> None:
    height, width = load_config("configs/experiment/default.yaml").supernet.input_resolutions["tiny"]
    image = Image.fromarray(np.zeros((1024, 2048, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((1024, 2048), dtype=np.uint8))
    tensor, target = SegmentationResizeToTensor(height=height, width=width)(image, mask)
    assert tuple(tensor.shape) == (3, 192, 384)
    assert tuple(target.shape) == (192, 384)
