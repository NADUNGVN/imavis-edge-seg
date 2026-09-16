"""Evaluate a trained supernet checkpoint: mIoU per elasticity level, on Cityscapes val
and on ACDC val (overall + per adverse condition), per RESEARCH_PLAN.md §8.

    uv run python scripts/evaluate_supernet.py --checkpoint outputs/<exp>/checkpoints/step_00002000.pt --config configs/experiment/default.yaml

Levels/datasets can be restricted with --level/--dataset for a faster partial check.
Add --per-class to also record per-class IoU and valid (non-ignored) pixel count per
split -- needed to tell a real accuracy difference between two splits (e.g. a
condition mIoU that looks surprisingly high) apart from one split simply having a
different ignore-pixel fraction or class mix, which plain mIoU alone cannot show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from imavis_edge_seg.config import ElasticityLevel, ExperimentConfig, load_config
from imavis_edge_seg.data.acdc import ALL_CONDITIONS
from imavis_edge_seg.evaluation.data import build_acdc_eval_loader, build_cityscapes_eval_loader
from imavis_edge_seg.evaluation.evaluator import evaluate_level
from imavis_edge_seg.evaluation.metrics import EvalResult
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.training.checkpoint import load_checkpoint
from imavis_edge_seg.training.quantization import apply_qat


def _dataset_root(config: ExperimentConfig, name: str) -> Path | None:
    for dataset_config in config.datasets:
        if dataset_config.name == name:
            return dataset_config.root
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    parser.add_argument("--level", action="append", default=[], help="restrict to these levels (repeatable)")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        choices=["cityscapes", "acdc"],
        help="restrict to these datasets (repeatable)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="also record per-class IoU and valid-pixel count per split (diagnostic, "
        "changes --output-json's shape from {level: {dataset: miou}} to "
        "{level: {dataset: {miou, num_pixels, per_class_iou}}})",
    )
    parser.add_argument(
        "--qat",
        action="store_true",
        help="apply_qat before loading the checkpoint -- required to correctly "
        "evaluate a checkpoint that was *trained* with --qat (train_supernet.py). "
        "Loading such a checkpoint into a plain, non-QAT supernet would evaluate its "
        "QAT-trained weights at full FP32 precision, not the INT8 fake-quantized "
        "inference the checkpoint was actually trained/is meant to be deployed under.",
    )
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config)
    supernet = PaceSegSupernet(config.supernet)
    if args.qat:
        supernet = apply_qat(supernet)  # type: ignore[assignment]
    supernet = supernet.to(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
    supernet.load_state_dict(checkpoint["model_state_dict"])
    console.print(
        f"loaded checkpoint step={checkpoint['step']} config_hash={checkpoint['config_hash']} "
        f"git_commit={checkpoint['git_commit']} qat={args.qat}"
    )

    levels: list[ElasticityLevel] = args.level or list(config.supernet.levels)
    datasets = args.dataset or ["cityscapes", "acdc"]
    results: dict[str, dict[str, object]] = {}

    table = Table(title=f"mIoU -- {args.checkpoint}")
    table.add_column("level")
    table.add_column("dataset")
    table.add_column("mIoU", justify="right")
    if args.per_class:
        table.add_column("valid px", justify="right")

    per_class_table = Table(title="per-class IoU") if args.per_class else None
    if per_class_table is not None:
        per_class_table.add_column("level/dataset")

    cityscapes_root = _dataset_root(config, "cityscapes")
    acdc_root = _dataset_root(config, "acdc")

    def _record(level: ElasticityLevel, split_name: str, result: EvalResult) -> None:
        if args.per_class:
            results[level][split_name] = {
                "miou": result.miou,
                "num_pixels": result.num_pixels,
                "per_class_iou": result.per_class_iou,
            }
            table.add_row(level, split_name, f"{result.miou:.4f}", f"{result.num_pixels:,}")
            if per_class_table is not None:
                if not per_class_table.columns[1:]:
                    for class_name in result.per_class_iou:
                        per_class_table.add_column(class_name, justify="right")
                per_class_table.add_row(
                    f"{level}/{split_name}",
                    *(f"{v:.3f}" if v == v else "nan" for v in result.per_class_iou.values()),
                )
        else:
            results[level][split_name] = result.miou
            table.add_row(level, split_name, f"{result.miou:.4f}")

    for level in levels:
        results[level] = {}

        if "cityscapes" in datasets and cityscapes_root is not None:
            loader = build_cityscapes_eval_loader(
                config, level, cityscapes_root, split="val", batch_size=args.batch_size
            )
            result = evaluate_level(supernet, level, loader, device=args.device)
            _record(level, "cityscapes", result)

        if "acdc" in datasets and acdc_root is not None:
            for condition in ALL_CONDITIONS:
                loader = build_acdc_eval_loader(
                    config, level, acdc_root, condition, split="val", batch_size=args.batch_size
                )
                result = evaluate_level(supernet, level, loader, device=args.device)
                _record(level, f"acdc/{condition}", result)

    console.print(table)
    if per_class_table is not None:
        console.print(per_class_table)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        console.print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
