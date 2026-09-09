"""Joint image+mask transforms. Deliberately minimal: resize to a fixed
(height, width) -- the same static-shape requirement the deployment side needs (see
`docs/RESEARCH_PLAN.md` §5.3) -- plus tensor conversion and ImageNet-style
normalization. No augmentation here; that belongs to the training loop, not the
dataset, so the same transform can be reused for train/val/test.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from imavis_edge_seg.data.labels import IGNORE_INDEX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationResizeToTensor:
    """Resize (image: bilinear, mask: nearest) to `(height, width)`, then convert to
    normalized float tensors. `height`/`width` should match one of
    `SupernetConfig.input_resolutions` for the elasticity level being trained."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        image = image.convert("RGB").resize(
            (self.width, self.height), Image.Resampling.BILINEAR
        )
        mask = mask.resize((self.width, self.height), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        for channel in range(3):
            image_array[:, :, channel] = (
                image_array[:, :, channel] - IMAGENET_MEAN[channel]
            ) / IMAGENET_STD[channel]
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()

        mask_array = np.asarray(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array).long()
        mask_tensor[(mask_tensor < 0) | (mask_tensor > 18)] = IGNORE_INDEX

        return image_tensor, mask_tensor
