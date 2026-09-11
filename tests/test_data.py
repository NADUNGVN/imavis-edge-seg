from pathlib import Path

import numpy as np
import pytest

from imavis_edge_seg.data.labels import IGNORE_INDEX, NUM_CLASSES, id_mask_to_train_id
from imavis_edge_seg.data.manifest import (
    build_manifest,
    discover_acdc_samples,
    discover_cityscapes_samples,
    read_manifest_csv,
    write_manifest_csv,
)

Image = pytest.importorskip("PIL.Image")


def test_id_mask_to_train_id_known_values() -> None:
    raw = np.array([[7, 24, 0, 255], [26, 33, 1, 9]], dtype=np.uint8)
    train_ids = id_mask_to_train_id(raw)
    assert train_ids[0, 0] == 0  # road
    assert train_ids[0, 1] == 11  # person
    assert train_ids[0, 2] == IGNORE_INDEX  # unlabeled
    assert train_ids[0, 3] == IGNORE_INDEX  # license plate / out of range
    assert train_ids[1, 0] == 13  # car
    assert train_ids[1, 1] == 18  # bicycle
    assert train_ids[1, 2] == IGNORE_INDEX  # ego vehicle
    assert train_ids[1, 3] == IGNORE_INDEX  # parking, not a trainId class


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.fixture
def fake_cityscapes(tmp_path: Path) -> Path:
    root = tmp_path / "cityscapes"
    rgb = (np.random.rand(32, 64, 3) * 255).astype(np.uint8)
    label_ids = np.full((32, 64), 7, dtype=np.uint8)  # all "road"
    _write_png(root / "leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png", rgb)
    _write_png(root / "gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png", label_ids)
    return root


@pytest.fixture
def fake_acdc(tmp_path: Path) -> Path:
    root = tmp_path / "acdc"
    rgb = (np.random.rand(32, 64, 3) * 255).astype(np.uint8)
    train_ids = np.full((32, 64), 10, dtype=np.uint8)  # all "sky"
    _write_png(root / "rgb_anon/fog/train/GOPR0001/GOPR0001_frame_000001_rgb_anon.png", rgb)
    _write_png(root / "gt/fog/train/GOPR0001/GOPR0001_frame_000001_gt_labelTrainIds.png", train_ids)
    return root


def test_discover_cityscapes_samples(fake_cityscapes: Path) -> None:
    samples = discover_cityscapes_samples(fake_cityscapes, "train")
    assert len(samples) == 1
    image_path, label_path = samples[0]
    assert image_path.name == "aachen_000000_000019_leftImg8bit.png"
    assert label_path.name == "aachen_000000_000019_gtFine_labelIds.png"


def test_discover_cityscapes_samples_missing_split(fake_cityscapes: Path) -> None:
    assert discover_cityscapes_samples(fake_cityscapes, "val") == []


def test_discover_acdc_samples(fake_acdc: Path) -> None:
    samples = discover_acdc_samples(fake_acdc, "train", ("fog", "night", "rain", "snow"))
    assert len(samples) == 1
    samples_fog_only = discover_acdc_samples(fake_acdc, "train", ("fog",))
    assert len(samples_fog_only) == 1
    samples_no_fog = discover_acdc_samples(fake_acdc, "train", ("night",))
    assert samples_no_fog == []


def test_manifest_roundtrip(fake_cityscapes: Path, tmp_path: Path) -> None:
    samples = discover_cityscapes_samples(fake_cityscapes, "train")
    rows = build_manifest(fake_cityscapes, samples, compute_checksums=True)
    assert len(rows) == 1
    assert len(rows[0].image_sha256) == 64  # sha256 hex digest length

    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(rows, manifest_path)
    reloaded = read_manifest_csv(manifest_path)
    assert reloaded == rows


def test_manifest_without_checksums(fake_cityscapes: Path) -> None:
    samples = discover_cityscapes_samples(fake_cityscapes, "train")
    rows = build_manifest(fake_cityscapes, samples, compute_checksums=False)
    assert rows[0].image_sha256 == ""


torch = pytest.importorskip("torch")

from imavis_edge_seg.data.cityscapes import CityscapesDataset  # noqa: E402
from imavis_edge_seg.data.transforms import (  # noqa: E402
    SegmentationResizeToTensor,
    SegmentationTrainAugment,
)


def test_cityscapes_dataset_loads_and_resizes(fake_cityscapes: Path) -> None:
    transform = SegmentationResizeToTensor(height=16, width=32)
    dataset = CityscapesDataset(fake_cityscapes, split="train", transform=transform)
    assert len(dataset) == 1
    image, mask = dataset[0]
    assert image.shape == (3, 16, 32)
    assert mask.shape == (16, 32)
    assert torch.all(mask == 0)  # constant "road" mask, trainId 0
    assert int(mask.max()) < NUM_CLASSES


def test_cityscapes_dataset_raises_on_empty_split(fake_cityscapes: Path) -> None:
    transform = SegmentationResizeToTensor(height=16, width=32)
    with pytest.raises(FileNotFoundError):
        CityscapesDataset(fake_cityscapes, split="val", transform=transform)


def _random_image_and_mask(height: int, width: int) -> tuple["Image.Image", "Image.Image"]:
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), mode="RGB")
    mask = Image.fromarray(rng.integers(0, NUM_CLASSES, (height, width), dtype=np.uint8), mode="L")
    return image, mask


def test_segmentation_train_augment_output_shape_always_matches_target() -> None:
    transform = SegmentationTrainAugment(height=32, width=64)
    image, mask = _random_image_and_mask(48, 96)
    for _ in range(10):  # random scale each call -- shape must stay fixed regardless
        image_tensor, mask_tensor = transform(image, mask)
        assert image_tensor.shape == (3, 32, 64)
        assert mask_tensor.shape == (32, 64)


def test_segmentation_train_augment_mask_values_stay_in_valid_range() -> None:
    transform = SegmentationTrainAugment(height=32, width=64)
    image, mask = _random_image_and_mask(48, 96)
    for _ in range(10):
        _, mask_tensor = transform(image, mask)
        valid = (mask_tensor >= 0) & (mask_tensor < NUM_CLASSES)
        assert bool(((mask_tensor == IGNORE_INDEX) | valid).all())


def test_segmentation_train_augment_downscale_pads_with_ignore_index() -> None:
    # scale_range fixed < 1 guarantees the scaled image is smaller than the target,
    # so padding (and therefore IGNORE_INDEX pixels) must appear in the output.
    transform = SegmentationTrainAugment(height=32, width=64, scale_range=(0.5, 0.5))
    image, mask = _random_image_and_mask(32, 64)
    _, mask_tensor = transform(image, mask)
    assert bool((mask_tensor == IGNORE_INDEX).any())


def test_segmentation_train_augment_is_actually_random() -> None:
    transform = SegmentationTrainAugment(height=32, width=64)
    image, mask = _random_image_and_mask(48, 96)
    outputs = [transform(image, mask)[0] for _ in range(8)]
    assert not all(torch.equal(outputs[0], out) for out in outputs[1:])
