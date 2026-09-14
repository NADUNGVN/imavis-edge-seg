"""Evaluate a trained required-baseline checkpoint (RESEARCH_PLAN.md §7): mIoU on
Cityscapes val and on ACDC val (overall + per adverse condition), at the same
resolution it was trained at (the supernet's largest configured level's resolution --
`training.baseline_trainer` reuses `training.data.build_train_dataloader`, which always
loads at that resolution).

    uv run python scripts/evaluate_baseline.py --model mobilenetv3_deeplabv3 --checkpoint outputs/baseline_mobilenetv3_deeplabv3_seed0/checkpoints/step_00100000.pt --config configs/experiment/default.yaml

Add --per-class for the same per-class IoU / valid-pixel diagnostic evaluate_supernet.py
has -- needed to compare against a supernet level's per-class breakdown properly rather
than aggregate mIoU alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from imavis_edge_seg.config import ExperimentConfig, load_config
from imavis_edge_seg.data.acdc import ALL_CONDITIONS
from imavis_edge_seg.evaluation.data import build_acdc_eval_loader, build_cityscapes_eval_loader
from imavis_edge_seg.evaluation.metrics import ConfusionMatrixAccumulator, EvalResult
from imavis_edge_seg.models.baselines import BASELINE_NAMES, build_baseline_model
from imavis_edge_seg.training.checkpoint import load_checkpoint
from imavis_edge_seg.training.quantization import apply_qat


def _dataset_root(config: ExperimentConfig, name: str) -> Path | None:
    for dataset_config in config.datasets:
        if dataset_config.name == name:
            return dataset_config.root
    return None


@torch.no_grad()
def _evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str) -> EvalResult:  # type: ignore[type-arg]
    model.eval()
    accumulator = ConfusionMatrixAccumulator()
    for image, mask in dataloader:
        image = image.to(device)
        mask = mask.to(device)
        logits = model(image)
        pred = logits.argmax(dim=1)
        accumulator.update(pred, mask)
    return accumulator.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=BASELINE_NAMES)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    parser.add_argument(
        "--dataset", action="append", default=[], choices=["cityscapes", "acdc"],
        help="restrict to these datasets (repeatable)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--per-class", action="store_true", help="see evaluate_supernet.py --per-class")
    parser.add_argument(
        "--qat",
        action="store_true",
        help="apply_qat before loading the checkpoint -- required to correctly "
        "evaluate a checkpoint that was *trained* with --qat (train_baseline.py). "
        "Loading such a checkpoint into a plain, non-QAT model would evaluate its "
        "QAT-trained weights at full FP32 precision, not the INT8 fake-quantized "
        "inference the checkpoint was actually trained/is meant to be deployed under.",
    )
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config)
    model = build_baseline_model(args.model, num_classes=config.supernet.num_classes)
    if args.qat:
        model = apply_qat(model)
    model = model.to(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    console.print(
        f"model={args.model} qat={args.qat} loaded checkpoint step={checkpoint['step']} "
        f"config_hash={checkpoint['config_hash']} git_commit={checkpoint['git_commit']}"
    )

    # Evaluated at the supernet's largest configured level's resolution -- the same
    # resolution training.data.build_train_dataloader used for this baseline's training.
    eval_level = config.supernet.levels[-1]
    datasets = args.dataset or ["cityscapes", "acdc"]
    results: dict[str, object] = {}

    table = Table(title=f"mIoU -- {args.model} -- {args.checkpoint}")
    table.add_column("dataset")
    table.add_column("mIoU", justify="right")
    if args.per_class:
        table.add_column("valid px", justify="right")

    per_class_table = Table(title="per-class IoU") if args.per_class else None
    if per_class_table is not None:
        per_class_table.add_column("dataset")

    cityscapes_root = _dataset_root(config, "cityscapes")
    acdc_root = _dataset_root(config, "acdc")

    def _record(split_name: str, result: EvalResult) -> None:
        if args.per_class:
            results[split_name] = {
                "miou": result.miou,
                "num_pixels": result.num_pixels,
                "per_class_iou": result.per_class_iou,
            }
            table.add_row(split_name, f"{result.miou:.4f}", f"{result.num_pixels:,}")
            if per_class_table is not None:
                if not per_class_table.columns[1:]:
                    for class_name in result.per_class_iou:
                        per_class_table.add_column(class_name, justify="right")
                per_class_table.add_row(
                    split_name,
                    *(f"{v:.3f}" if v == v else "nan" for v in result.per_class_iou.values()),
                )
        else:
            results[split_name] = result.miou
            table.add_row(split_name, f"{result.miou:.4f}")

    if "cityscapes" in datasets and cityscapes_root is not None:
        loader = build_cityscapes_eval_loader(
            config, eval_level, cityscapes_root, split="val", batch_size=args.batch_size
        )
        _record("cityscapes", _evaluate(model, loader, args.device))

    if "acdc" in datasets and acdc_root is not None:
        for condition in ALL_CONDITIONS:
            loader = build_acdc_eval_loader(
                config, eval_level, acdc_root, condition, split="val", batch_size=args.batch_size
            )
            _record(f"acdc/{condition}", _evaluate(model, loader, args.device))

    console.print(table)
    if per_class_table is not None:
        console.print(per_class_table)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        console.print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
