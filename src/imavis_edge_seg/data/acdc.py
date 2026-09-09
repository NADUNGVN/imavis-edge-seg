"""ACDC (Adverse Conditions Dataset with Correspondences) loader.

Expects the official on-disk layout (`{type}/{condition}/{split}/{scene}/...` per
ACDC's own README -- condition outside split, verified against a real extracted
download 2026-09-09, not assumed):

    <root>/rgb_anon/<condition>/<split>/<scene>/<name>_rgb_anon.png
    <root>/gt/<condition>/<split>/<scene>/<name>_gt_labelTrainIds.png

Unlike Cityscapes, ACDC ships `labelTrainIds` masks directly, already in the same
19-class scheme as `imavis_edge_seg.data.labels` -- no id-to-trainId conversion needed.
The official test split has no public labels (submit to ACDC's evaluation server for
test-set numbers); this loader only supports `train`/`val` labels. See
`docs/DATASET.md` for how to obtain the dataset (registration required) and dataset
scope policy (ACDC is the primary adverse-condition dataset per
`docs/RESEARCH_PLAN.md` §6; Dark Zurich is external validation only, never tuned on).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from imavis_edge_seg.data.manifest import discover_acdc_samples
from imavis_edge_seg.data.transforms import SegmentationResizeToTensor

ACDCSplit = Literal["train", "val"]
ACDCCondition = Literal["fog", "night", "rain", "snow"]
ALL_CONDITIONS: tuple[ACDCCondition, ...] = ("fog", "night", "rain", "snow")


class ACDCDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        root: str | Path,
        split: ACDCSplit,
        transform: SegmentationResizeToTensor,
        conditions: tuple[ACDCCondition, ...] = ALL_CONDITIONS,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.conditions = conditions
        self.samples = discover_acdc_samples(self.root, split, conditions)
        if not self.samples:
            raise FileNotFoundError(
                f"No ACDC samples found under {self.root} for split={split!r}, "
                f"conditions={conditions!r} -- check the directory layout in "
                "docs/DATASET.md"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path)
        mask = Image.open(label_path)
        return self.transform(image, mask)
