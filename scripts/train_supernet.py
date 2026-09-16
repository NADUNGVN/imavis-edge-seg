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
    parser.add_argument(
        "--qat",
        action="store_true",
        help="apply QAT (training/quantization.py::apply_qat) before training -- fake-"
        "quantizes every nn.Conv2d and SlimmableConv2d's weight and input to INT8 "
        "during the whole run",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="load an existing (typically FP32) supernet checkpoint's weights before "
        "training/QAT starts -- RESEARCH_PLAN.md §5.2's 'FP32 teacher -> QAT INT8' "
        "workflow. Use a distinct --override experiment_id=... so this run's own "
        "checkpoints don't land in the FP32 run's directory.",
    )
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config, overrides=args.override or None)
    console.print(
        f"experiment_id={config.experiment_id} config_hash={config.config_hash()} "
        f"device={args.device} qat={args.qat}"
    )

    dataloader = build_train_dataloader(config)
    console.print(f"train dataset size: {len(dataloader.dataset)}")  # type: ignore[arg-type]

    output_dir = config.output_root / config.experiment_id
    run_training(
        config,
        dataloader,
        output_dir,
        console=console,
        device=args.device,
        qat=args.qat,
        init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
