"""Cityscapes 19-class palette helpers shared by the V20 figure scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                    (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)],
                   dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), np.uint8)
    ok = mask < len(PALETTE)
    out[ok] = PALETTE[mask[ok]]
    return out


def colorize_png(src: Path, dst: Path) -> None:
    Image.fromarray(colorize(np.asarray(Image.open(src)))).save(dst)
