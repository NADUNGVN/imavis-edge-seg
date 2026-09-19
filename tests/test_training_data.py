from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from imavis_edge_seg.config import DatasetConfig, ExperimentConfig  # noqa: E402
from imavis_edge_seg.training.data import (  # noqa: E402
    build_calibration_dataloader,
    build_train_dataloader,
)


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.fixture
def fake_cityscapes(tmp_path: Path) -> Path:
    root = tmp_path / "cityscapes"
    rgb = (np.random.rand(32, 64, 3) * 255).astype(np.uint8)
    label_ids = np.full((32, 64), 7, dtype=np.uint8)  # all "road"
    for i in range(6):
        _write_png(root / f"leftImg8bit/train/city/city_{i:06d}_000019_leftImg8bit.png", rgb)
        _write_png(root / f"gtFine/train/city/city_{i:06d}_000019_gtFine_labelIds.png", label_ids)
    return root


def _tiny_config(cityscapes_root: Path) -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_id="test-calibration",
        datasets=[DatasetConfig(name="cityscapes", root=cityscapes_root, split="train")],
    )
    for level in config.supernet.levels:
        config.supernet.input_resolutions[level] = (32, 64)
    config.training.augment = True  # deliberately on, to check calibration forces it off
    config.training.batch_size = 2
    return config


def test_build_calibration_dataloader_disables_augmentation_regardless_of_config(
    fake_cityscapes: Path,
) -> None:
    config = _tiny_config(fake_cityscapes)
    loader = build_calibration_dataloader(config, max_images=6)
    assert config.training.augment is True  # the caller's config object is untouched

    from imavis_edge_seg.data.transforms import SegmentationTrainAugment

    assert not isinstance(loader.dataset.transform, SegmentationTrainAugment)  # type: ignore[attr-defined]


def test_build_calibration_dataloader_subsamples_to_max_images(fake_cityscapes: Path) -> None:
    config = _tiny_config(fake_cityscapes)
    loader = build_calibration_dataloader(config, max_images=3)
    assert len(loader.dataset) == 3  # type: ignore[arg-type]


def test_build_calibration_dataloader_keeps_all_images_under_the_cap(fake_cityscapes: Path) -> None:
    config = _tiny_config(fake_cityscapes)
    loader = build_calibration_dataloader(config, max_images=200)
    assert len(loader.dataset) == 6  # type: ignore[arg-type]


def test_build_calibration_dataloader_yields_real_batches(fake_cityscapes: Path) -> None:
    config = _tiny_config(fake_cityscapes)
    loader = build_calibration_dataloader(config, max_images=4)
    image, mask = next(iter(loader))
    assert image.shape[0] == config.training.batch_size
    assert image.shape[-2:] == (32, 64)
    assert mask.shape[-2:] == (32, 64)


def test_build_train_dataloader_defaults_to_largest_level_resolution(fake_cityscapes: Path) -> None:
    config = _tiny_config(fake_cityscapes)
    config.supernet.input_resolutions["tiny"] = (16, 32)
    config.supernet.input_resolutions[config.supernet.levels[-1]] = (32, 64)
    loader = build_train_dataloader(config)
    image, _mask = next(iter(loader))
    assert image.shape[-2:] == (32, 64)


def test_build_train_dataloader_level_overrides_resolution(fake_cityscapes: Path) -> None:
    config = _tiny_config(fake_cityscapes)
    config.supernet.input_resolutions["tiny"] = (16, 32)
    loader = build_train_dataloader(config, level="tiny")  # type: ignore[arg-type]
    image, mask = next(iter(loader))
    assert image.shape[-2:] == (16, 32)
    assert mask.shape[-2:] == (16, 32)


def test_build_calibration_dataloader_level_overrides_resolution(fake_cityscapes: Path) -> None:
    config = _tiny_config(fake_cityscapes)
    config.supernet.input_resolutions["tiny"] = (16, 32)
    loader = build_calibration_dataloader(config, max_images=4, level="tiny")  # type: ignore[arg-type]
    image, _mask = next(iter(loader))
    assert image.shape[-2:] == (16, 32)
