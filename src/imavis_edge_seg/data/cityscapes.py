"""Cityscapes dataset loader.

Expects the standard on-disk layout from the official download (`leftImg8bit_*.zip` +
`gtFine_trainvaltest.zip`, extracted side by side under one root):

    <root>/leftImg8bit/<split>/<city>/<city>_<seq>_<frame>_leftImg8bit.png
    <root>/gtFine/<split>/<city>/<city>_<seq>_<frame>_gtFine_labelIds.png

Cityscapes does not ship `trainId` masks directly -- `id_mask_to_train_id` converts the
raw `labelIds` mask on the fly. See `docs/DATASET.md` for how to obtain the dataset
(registration required) and the exact directory layout expected here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from imavis_edge_seg.data.labels import id_mask_to_train_id
from imavis_edge_seg.data.manifest import discover_cityscapes_samples
from imavis_edge_seg.data.transforms import SegmentationResizeToTensor

CityscapesSplit = Literal["train", "val", "test"]


class CityscapesDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        root: str | Path,
        split: CityscapesSplit,
        transform: SegmentationResizeToTensor,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = discover_cityscapes_samples(self.root, split)
        if not self.samples:
            raise FileNotFoundError(
                f"No Cityscapes samples found under {self.root} for split={split!r} "
                "-- check the directory layout in docs/DATASET.md"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path)
        raw_mask = np.array(Image.open(label_path), dtype=np.uint8)
        train_id_mask = Image.fromarray(id_mask_to_train_id(raw_mask))
        return self.transform(image, train_id_mask)
