"""Builds the training DataLoader from `ExperimentConfig.datasets`. Every dataset is
loaded at the *largest* elasticity level's resolution -- `training.step.train_step`
downsamples per-level from there, so the loader itself never needs to know about the
sandwich rule.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from imavis_edge_seg.config import ExperimentConfig
from imavis_edge_seg.data.acdc import ACDCDataset
from imavis_edge_seg.data.cityscapes import CityscapesDataset
from imavis_edge_seg.data.transforms import SegmentationResizeToTensor

_Sample = tuple[Tensor, Tensor]


def build_train_dataset(config: ExperimentConfig) -> Dataset[_Sample]:
    largest_level = config.supernet.levels[-1]
    height, width = config.supernet.input_resolutions[largest_level]
    transform = SegmentationResizeToTensor(height=height, width=width)

    datasets: list[Dataset[_Sample]] = []
    for dataset_config in config.datasets:
        if dataset_config.name == "cityscapes":
            datasets.append(
                CityscapesDataset(dataset_config.root, split=dataset_config.split, transform=transform)  # type: ignore[arg-type]
            )
        elif dataset_config.name == "acdc":
            datasets.append(
                ACDCDataset(dataset_config.root, split=dataset_config.split, transform=transform)  # type: ignore[arg-type]
            )
        else:
            raise ValueError(
                f"Dataset {dataset_config.name!r} has no training loader yet "
                "(dark_zurich/bdd100k are external-validation/optional, see docs/RESEARCH_PLAN.md §6)"
            )

    if not datasets:
        raise ValueError("config.datasets is empty -- nothing to train on")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_train_dataloader(config: ExperimentConfig) -> DataLoader[_Sample]:
    dataset = build_train_dataset(config)
    # `training.trainer`/`training.baseline_trainer` both loop training "forever" by
    # re-iterating this same DataLoader (`for _ in itertools.count(): yield from
    # loader`) rather than re-creating it -- without persistent_workers, every epoch
    # boundary (here, every ~len(dataset)/batch_size steps -- a few hundred, given a
    # ~4.5k-image dataset) tears down and respawns all worker processes, starving the
    # GPU for a moment each time. Observed live as uneven MBW/GPU-util in nvitop during
    # a baseline run. persistent_workers requires num_workers > 0.
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.training.num_workers > 0,
    )
