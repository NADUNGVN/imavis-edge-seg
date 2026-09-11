"""Joint image+mask transforms.

`SegmentationResizeToTensor` stays deterministic (resize to a fixed `(height, width)` --
the same static-shape requirement the deployment side needs, see
`docs/RESEARCH_PLAN.md` §5.3 -- plus tensor conversion and ImageNet normalization) so
the exact same transform is safe to reuse for val/test, where reproducible numbers
matter.

`SegmentationTrainAugment` is training-only: random scale + crop, horizontal flip and
color jitter, *then* the same fixed-size resize + normalize as
`SegmentationResizeToTensor` -- augmentation varies what the model sees during
training, it does not change the static `(height, width)` the exported model graph
commits to. Added 2026-09-11: 100k training steps over a ~4.5k-image dataset with zero
augmentation was flagged as the single highest-leverage remaining gap in
`reports/first_full_supernet_run_100k_20260910.md`.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import ColorJitter

from imavis_edge_seg.data.labels import IGNORE_INDEX

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_normalized_tensors(image: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
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
        return _to_normalized_tensors(image, mask)


class SegmentationTrainAugment:
    """Training-time augmentation, output still a fixed `(height, width)`:

    1. Random scale in `scale_range` (relative to the target size), image bilinear /
       mask nearest, so the same random factor cannot desync the two.
    2. Pad (image: 0, mask: `IGNORE_INDEX` -- never count padded pixels toward loss)
       if the scaled image is smaller than the target in either dimension.
    3. Random crop to exactly `(height, width)`.
    4. Random horizontal flip with probability `hflip_prob`, applied identically to
       both image and mask.
    5. Color jitter (image only -- a class label doesn't have a "brightness").
    6. The same ImageNet normalization `SegmentationResizeToTensor` uses.
    """

    def __init__(
        self,
        height: int,
        width: int,
        scale_range: tuple[float, float] = (0.75, 1.25),
        hflip_prob: float = 0.5,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
    ) -> None:
        self.height = height
        self.width = width
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.color_jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        image = image.convert("RGB")

        scale = random.uniform(*self.scale_range)
        scaled_h = max(round(self.height * scale), 1)
        scaled_w = max(round(self.width * scale), 1)
        image = image.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
        mask = mask.resize((scaled_w, scaled_h), Image.Resampling.NEAREST)

        pad_h = max(self.height - scaled_h, 0)
        pad_w = max(self.width - scaled_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=IGNORE_INDEX)
            scaled_h, scaled_w = scaled_h + pad_h, scaled_w + pad_w

        max_x = scaled_w - self.width
        max_y = scaled_h - self.height
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        image = image.crop((x, y, x + self.width, y + self.height))
        mask = mask.crop((x, y, x + self.width, y + self.height))

        if random.random() < self.hflip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image = self.color_jitter(image)

        return _to_normalized_tensors(image, mask)
