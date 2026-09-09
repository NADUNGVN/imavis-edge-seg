"""Launch supernet training from an experiment config.

    uv run python scripts/train_supernet.py --config configs/experiment/default.yaml

For a real (multi-hour+) run on a server, use `start_train.sh` (not yet added -- see
docs/COLLABORATION_PROTOCOL.md "Detached long-running jobs") instead of running this
directly in an interactive shell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console

from imavis_edge_seg.config import load_config
from imavis_edge_seg.training.data import build_train_dataloader
from imavis_edge_seg.training.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    parser.add_argument("--override", action="append", default=[], help="dotlist override, key=value")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config, overrides=args.override or None)
    console.print(f"experiment_id={config.experiment_id} config_hash={config.config_hash()} device={args.device}")

    dataloader = build_train_dataloader(config)
    console.print(f"train dataset size: {len(dataloader.dataset)}")  # type: ignore[arg-type]

    output_dir = config.output_root / config.experiment_id
    run_training(config, dataloader, output_dir, console=console, device=args.device)


if __name__ == "__main__":
    main()
