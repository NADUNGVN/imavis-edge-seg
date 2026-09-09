"""Checkpoint save/load with provenance (git commit, config hash, seed, step) per
`../../docs/SHARED_INFRASTRUCTURE.md` §1 rule 4 -- never silently overwrite a prior
experiment's output; every checkpoint records what produced it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from imavis_edge_seg.config import ExperimentConfig


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def save_checkpoint(
    path: Path,
    supernet: nn.Module,
    optimizer: Optimizer,
    step: int,
    config: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": supernet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.model_dump(mode="json"),
            "config_hash": config.config_hash(),
            "git_commit": _git_commit(),
            "seed": config.seed,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    checkpoint: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    return checkpoint
