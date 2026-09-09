"""Dataset manifests: a small, git-committable CSV recording which (image, label)
pairs exist, plus a checksum, so every training job reads the exact same file list
without re-scanning a possibly-huge raw data tree, and so the raw tree's identity is
provenance-tracked without committing the raw data itself (`docs/DATASET.md`,
`../../docs/SHARED_INFRASTRUCTURE.md` §1 rule 3-4).
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def discover_cityscapes_samples(
    root: Path, split: Literal["train", "val", "test"]
) -> list[tuple[Path, Path]]:
    """Pure filesystem discovery, no torch dependency -- shared by
    `CityscapesDataset` and the manifest-building CLI command."""
    image_root = root / "leftImg8bit" / split
    label_root = root / "gtFine" / split
    samples: list[tuple[Path, Path]] = []
    if not image_root.is_dir():
        return samples
    for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
        city = image_path.parent.name
        stem = image_path.name.removesuffix("_leftImg8bit.png")
        label_path = label_root / city / f"{stem}_gtFine_labelIds.png"
        if label_path.exists():
            samples.append((image_path, label_path))
    return samples


def discover_acdc_samples(
    root: Path, split: Literal["train", "val"], conditions: tuple[str, ...]
) -> list[tuple[Path, Path]]:
    """Pure filesystem discovery, no torch dependency -- shared by `ACDCDataset` and
    the manifest-building CLI command."""
    samples: list[tuple[Path, Path]] = []
    for condition in conditions:
        image_root = root / "rgb_anon" / split / condition
        label_root = root / "gt" / split / condition
        if not image_root.is_dir():
            continue
        for image_path in sorted(image_root.glob("*/*_rgb_anon.png")):
            scene = image_path.parent.name
            stem = image_path.name.removesuffix("_rgb_anon.png")
            label_path = label_root / scene / f"{stem}_gt_labelTrainIds.png"
            if label_path.exists():
                samples.append((image_path, label_path))
    return samples


@dataclass(frozen=True)
class ManifestRow:
    image_path: str  # relative to the dataset root
    label_path: str  # relative to the dataset root
    image_sha256: str
    label_sha256: str


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: Path, samples: list[tuple[Path, Path]], compute_checksums: bool = True
) -> list[ManifestRow]:
    """`samples` are absolute (image_path, label_path) pairs, as produced by
    `CityscapesDataset._discover_samples` / `ACDCDataset._discover_samples`."""
    rows = []
    for image_path, label_path in samples:
        image_hash = _sha256_file(image_path) if compute_checksums else ""
        label_hash = _sha256_file(label_path) if compute_checksums else ""
        rows.append(
            ManifestRow(
                image_path=str(image_path.relative_to(root)).replace("\\", "/"),
                label_path=str(label_path.relative_to(root)).replace("\\", "/"),
                image_sha256=image_hash,
                label_sha256=label_hash,
            )
        )
    return rows


def write_manifest_csv(rows: list[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label_path", "image_sha256", "label_sha256"])
        for row in rows:
            writer.writerow([row.image_path, row.label_path, row.image_sha256, row.label_sha256])


def read_manifest_csv(manifest_path: Path) -> list[ManifestRow]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            ManifestRow(
                image_path=r["image_path"],
                label_path=r["label_path"],
                image_sha256=r["image_sha256"],
                label_sha256=r["label_sha256"],
            )
            for r in reader
        ]
