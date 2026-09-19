"""Fine-tune QAT on a single extracted (static, independent-weight) subnet from a
trained FP32 supernet checkpoint -- the "exported-subnet" cells of the 2026-09-20 QAT
2x2 factorial screen (docs/COORDINATION_LOG.md, open thread #1): does QAT fine-tuning
on genuinely independent per-level weights recover the accuracy the shared-supernet
QAT cell lost, isolated from the dynamic-vs-calibrated activation-range axis (the
same --calibrate/--calibration-observer flags as train_baseline.py/train_supernet.py)?

    uv run python scripts/train_exported_subnet.py --level small \
        --fp32-checkpoint outputs/<fp32_run>/checkpoints/step_00002000.pt \
        --config configs/experiment/default.yaml --qat \
        --override experiment_id=qat_exported_small_dynamic_seed0

extract_subnet's output is left in eval() mode by construction
(StaticPaceSegSubnet.__init__ ends with self.eval()) -- explicitly re-.train()-ed
below before handing it to run_baseline_training, or BatchNorm running stats would
never update during fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
from rich.console import Console

from imavis_edge_seg.config import ElasticityLevel, load_config
from imavis_edge_seg.models import PaceSegSupernet, extract_subnet
from imavis_edge_seg.training.baseline_trainer import run_baseline_training
from imavis_edge_seg.training.checkpoint import load_checkpoint
from imavis_edge_seg.training.data import build_train_dataloader
from imavis_edge_seg.training.quantization import CalibrationObserver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True, choices=["tiny", "small", "medium", "large"])
    parser.add_argument(
        "--fp32-checkpoint",
        type=Path,
        required=True,
        help="a trained (non-QAT) supernet checkpoint to extract the subnet's weights from",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    parser.add_argument("--override", action="append", default=[], help="dotlist override, key=value")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--qat",
        action="store_true",
        help="apply QAT to the extracted subnet before fine-tuning -- this script's "
        "whole purpose, so normally always passed; kept as an explicit flag (not "
        "implicit) so an FP32-only fine-tune sanity check is still possible.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="fit calibrated (not dynamic) activation quantization ranges before "
        "training, from a real calibration set at this level's resolution. Requires --qat.",
    )
    parser.add_argument(
        "--calibration-images",
        type=int,
        default=200,
        help="max images in the calibration set (default: 200)",
    )
    parser.add_argument(
        "--calibration-observer",
        choices=["max", "ema_percentile"],
        default="max",
        help="'max': hard running maximum (original). 'ema_percentile': per-call high "
        "percentile (--calibration-percentile) combined across calls via an EMA "
        "(--calibration-momentum) -- part of the 2026-09-20 QAT-rescue 2x2 screen.",
    )
    parser.add_argument(
        "--calibration-percentile",
        type=float,
        default=0.999,
        help="only used with --calibration-observer ema_percentile (default: 0.999)",
    )
    parser.add_argument(
        "--calibration-momentum",
        type=float,
        default=0.9,
        help="only used with --calibration-observer ema_percentile (default: 0.9)",
    )
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config, overrides=args.override or None)
    level = cast(ElasticityLevel, args.level)
    console.print(
        f"exported_subnet level={level} experiment_id={config.experiment_id} "
        f"config_hash={config.config_hash()} device={args.device} qat={args.qat} "
        f"calibrate={args.calibrate}"
    )

    supernet = PaceSegSupernet(config.supernet)
    fp32_state = load_checkpoint(args.fp32_checkpoint, map_location="cpu")
    supernet.load_state_dict(fp32_state["model_state_dict"])
    console.print(f"extracted {level} subnet from {args.fp32_checkpoint} (step={fp32_state['step']})")

    subnet = extract_subnet(supernet, level)
    subnet.train()

    dataloader = build_train_dataloader(config, level=level)
    console.print(f"train dataset size: {len(dataloader.dataset)}")  # type: ignore[arg-type]

    output_dir = config.output_root / config.experiment_id
    run_baseline_training(
        f"exported_subnet_{level}",
        config,
        dataloader,
        output_dir,
        console=console,
        device=args.device,
        qat=args.qat,
        model=subnet,
        calibrate=args.calibrate,
        calibration_images=args.calibration_images,
        calibration_observer=cast(CalibrationObserver, args.calibration_observer),
        calibration_percentile=args.calibration_percentile,
        calibration_momentum=args.calibration_momentum,
        calibration_level=level,
    )


if __name__ == "__main__":
    main()
