"""Builds evaluation DataLoaders. Unlike `training.data` (always loads at the largest
level's resolution, then downsamples per-level for the sandwich rule), evaluation loads
each dataset at *that specific level's own* resolution, since accuracy should be
measured at the resolution the subnet actually runs at.
"""

from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torch.utils.data import DataLoader

from imavis_edge_seg.config import ElasticityLevel, ExperimentConfig
from imavis_edge_seg.data.acdc import ACDCCondition, ACDCDataset
from imavis_edge_seg.data.cityscapes import CityscapesDataset
from imavis_edge_seg.data.transforms import SegmentationResizeToTensor

_Sample = tuple[Tensor, Tensor]


def build_cityscapes_eval_loader(
    config: ExperimentConfig,
    level: ElasticityLevel,
    root: str | Path,
    split: str = "val",
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader[_Sample]:
    height, width = config.supernet.input_resolutions[level]
    transform = SegmentationResizeToTensor(height=height, width=width)
    dataset = CityscapesDataset(root, split=split, transform=transform)  # type: ignore[arg-type]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def build_acdc_eval_loader(
    config: ExperimentConfig,
    level: ElasticityLevel,
    root: str | Path,
    condition: ACDCCondition,
    split: str = "val",
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader[_Sample]:
    height, width = config.supernet.input_resolutions[level]
    transform = SegmentationResizeToTensor(height=height, width=width)
    dataset = ACDCDataset(root, split=split, transform=transform, conditions=(condition,))  # type: ignore[arg-type]
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
