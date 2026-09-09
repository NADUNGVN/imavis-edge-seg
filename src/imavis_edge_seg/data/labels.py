"""Cityscapes-compatible label scheme, shared by Cityscapes and ACDC.

ACDC ships label masks already in `trainId` space (`*_gt_labelTrainIds.png`), so no
conversion is needed for it. Cityscapes ships only raw `id` masks
(`*_gtFine_labelIds.png`) plus polygon JSON; the `ID_TO_TRAIN_ID` table below is the
standard 34-class -> 19-class collapse used throughout the semantic segmentation
literature (originally from `cityscapesscripts/helpers/labels.py`, MIT licensed),
reproduced here so this project does not need `cityscapesScripts` as a dependency just
to remap label images.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

NUM_CLASSES = 19
IGNORE_INDEX = 255

# Raw Cityscapes `id` (0-33, plus -1 for the license-plate class) -> 19-class `trainId`.
# Anything not listed here (and -1) maps to IGNORE_INDEX.
ID_TO_TRAIN_ID: dict[int, int] = {
    7: 0,  # road
    8: 1,  # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

# trainId -> human-readable class name, for reports/confusion matrices.
TRAIN_ID_TO_NAME: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


def id_mask_to_train_id(id_mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a raw Cityscapes `labelIds` mask to `trainId` space. Values not in
    `ID_TO_TRAIN_ID` (including -1 and 255) become `IGNORE_INDEX`."""
    lookup = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for raw_id, train_id in ID_TO_TRAIN_ID.items():
        lookup[raw_id] = train_id
    clipped = np.clip(id_mask.astype(np.int32), 0, 255).astype(np.uint8)
    return lookup[clipped]
