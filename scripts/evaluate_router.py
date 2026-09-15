"""Wire the router (`src/imavis_edge_seg/router/`) into a real inference pass and
measure it end to end against real data (`RESEARCH_PLAN.md` Contribution 3 / RQ3),
closing the "no real (risk, error) data" and "not wired into evaluation" gaps in
`reports/router_v1_20260914.md`.

Splits each dataset split's validation images in half by alternating index (§5's
"calibration is fit on validation data only" -- and never on the same images a policy
is then scored against, or the score would be optimistic): the even-indexed half fits
a real `RiskCalibrator` (`compute_risk_score`/`compute_per_image_error` at one cheap
"probe" level, with ground truth); the odd-indexed half is then routed per-image under
each `RouterConfig.strategy` (`static_small`, `static_large`, `entropy`,
`calibrated_risk` -- `oracle` needs an `--eval-json` too, see below) using ONLY the
probe-level risk score (no ground truth, matching real inference), and every level's
prediction is actually run so the *achieved* mIoU under each strategy's real per-image
choices can be measured, not just each choice in isolation.

    uv run python scripts/evaluate_router.py --checkpoint outputs/pace_seg_v1/checkpoints/step_00100000.pt --config configs/experiment/default.yaml --lookup-table outputs/benchmark_lookup_table.csv --device-id E3 --backend tensorrt_gpu --dataset cityscapes

Add --eval-json (an `evaluate_supernet.py --output-json` file) to also include the
`oracle` strategy (a constant pick of whichever level has the best known aggregate
mIoU, RESEARCH_PLAN.md's documented upper-bound comparison, not a per-image decision).

`entropy`'s raw risk (softmax entropy in nats, typically ~0.3-3) and
`calibrated_risk`'s calibrated risk (an error probability, 0-1) are on different
scales -- each strategy's default `risk_target` is auto-picked from the fit-half's
mean *on that strategy's own scale* (not a single shared default), otherwise
`entropy`'s nats-scale score looks enormous next to an error-scale target and the
policy over-escalates to the most expensive level almost every time regardless of
the actual per-image signal -- caught on a real run (2026-09-15, E3/cityscapes)
where `entropy` was spending 96% of `static_large`'s latency for only a modest mIoU
gain over `calibrated_risk`, which used 7x less latency for the same effect size
per ms spent (see `reports/router_v1_20260914.md`).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import cast

import torch
from rich.console import Console
from rich.table import Table

from imavis_edge_seg.config import ElasticityLevel, ExperimentConfig, RouterConfig, load_config
from imavis_edge_seg.data.acdc import ALL_CONDITIONS
from imavis_edge_seg.evaluation.data import build_acdc_eval_loader, build_cityscapes_eval_loader
from imavis_edge_seg.evaluation.metrics import ConfusionMatrixAccumulator
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.router.calibrator import RiskCalibrator, fit_risk_calibrator
from imavis_edge_seg.router.observed_error import compute_per_image_error
from imavis_edge_seg.router.policy import select_level
from imavis_edge_seg.router.risk_probe import compute_risk_score
from imavis_edge_seg.search.pareto import ParetoPoint, build_pareto_points
from imavis_edge_seg.training.checkpoint import load_checkpoint

STRATEGIES = ["static_small", "static_large", "entropy", "calibrated_risk"]


def _dataset_root(config: ExperimentConfig, name: str) -> Path | None:
    for dataset_config in config.datasets:
        if dataset_config.name == name:
            return dataset_config.root
    return None


def _build_loaders(
    config: ExperimentConfig, split_name: str, cityscapes_root: Path | None, acdc_root: Path | None
) -> dict[ElasticityLevel, torch.utils.data.DataLoader]:  # type: ignore[type-arg]
    loaders: dict[ElasticityLevel, torch.utils.data.DataLoader] = {}  # type: ignore[type-arg]
    for level in config.supernet.levels:
        if split_name == "cityscapes":
            assert cityscapes_root is not None
            loaders[level] = build_cityscapes_eval_loader(config, level, cityscapes_root, split="val", batch_size=1)
        else:
            assert acdc_root is not None
            condition = split_name.split("/", 1)[1]
            loaders[level] = build_acdc_eval_loader(config, level, acdc_root, condition, split="val", batch_size=1)  # type: ignore[arg-type]
    return loaders


def _fallback_latency(level: ElasticityLevel) -> float:
    """Only used if `--lookup-table` has no row for a level on the chosen target --
    a synthetic monotonic placeholder so the routing demo still runs, clearly not a
    real measurement (a warning is printed whenever this is hit)."""
    return {"tiny": 1.0, "small": 2.0, "medium": 4.0, "large": 8.0}[level]


@torch.no_grad()
def _evaluate_split(
    console: Console,
    supernet: torch.nn.Module,
    config: ExperimentConfig,
    split_name: str,
    probe_level: ElasticityLevel,
    candidates: list[ParetoPoint],
    latency_by_level: dict[ElasticityLevel, float],
    error_risk_target_override: float | None,
    raw_risk_target_override: float | None,
    device: str,
    cityscapes_root: Path | None,
    acdc_root: Path | None,
    oracle_level: ElasticityLevel | None,
) -> dict[str, object]:
    loaders = _build_loaders(config, split_name, cityscapes_root, acdc_root)
    iterators = {level: iter(loader) for level, loader in loaders.items()}
    num_images = len(loaders[probe_level].dataset)  # type: ignore[arg-type]

    fit_scores: list[float] = []
    fit_errors: list[float] = []
    test_probe_risk: list[float] = []
    test_per_level: list[dict[ElasticityLevel, tuple[torch.Tensor, torch.Tensor]]] = []

    for i in range(num_images):
        batches = {level: next(iterators[level]) for level in loaders}
        is_fit = i % 2 == 0

        probe_image, probe_mask = batches[probe_level]
        probe_image, probe_mask = probe_image.to(device), probe_mask.to(device)
        probe_logits = supernet(probe_image, probe_level)
        probe_pred = probe_logits.argmax(dim=1)

        if is_fit:
            fit_scores.append(float(compute_risk_score(probe_logits, probe_mask)))
            fit_errors.append(float(compute_per_image_error(probe_pred, probe_mask)))
        else:
            test_probe_risk.append(float(compute_risk_score(probe_logits)))
            per_level: dict[ElasticityLevel, tuple[torch.Tensor, torch.Tensor]] = {}
            for level, (image, mask) in batches.items():
                image, mask = image.to(device), mask.to(device)
                logits = probe_logits if level == probe_level else supernet(image, level)
                pred = logits.argmax(dim=1)
                per_level[level] = (pred.cpu(), mask.cpu())
            test_per_level.append(per_level)

    if not fit_scores or not test_probe_risk:
        raise ValueError(f"{split_name}: need at least 2 val images (got {num_images}) to split fit/test")

    calibrator = fit_risk_calibrator(fit_scores, fit_errors)
    # entropy's raw_risk (nats, unbounded, typically ~0.3-3) and calibrated_risk's
    # calibrated_risk (an error probability, 0-1) live on different scales -- each
    # needs its *own* risk_target on its *own* scale (independently overridable), or
    # a shared target makes one strategy's threshold meaningless on the other's
    # scale. Defaults: calibrated_risk from the fit-half's mean observed error,
    # entropy from the fit-half's mean raw score.
    error_risk_target = error_risk_target_override if error_risk_target_override is not None else mean(fit_errors)
    raw_risk_target = raw_risk_target_override if raw_risk_target_override is not None else mean(fit_scores)

    strategies = list(STRATEGIES)
    if oracle_level is not None:
        strategies.append("oracle")

    strategy_results: dict[str, object] = {}
    for strategy in strategies:
        accumulator = ConfusionMatrixAccumulator()
        chosen_latencies: list[float] = []
        for raw_risk, per_level in zip(test_probe_risk, test_per_level, strict=True):
            if strategy == "oracle":
                chosen = oracle_level
            elif strategy == "entropy":
                router_config = RouterConfig(strategy=strategy, risk_target=raw_risk_target)  # type: ignore[arg-type]
                chosen = select_level(candidates, router_config, raw_risk=raw_risk)
            elif strategy == "calibrated_risk":
                router_config = RouterConfig(strategy=strategy, risk_target=error_risk_target)  # type: ignore[arg-type]
                chosen = select_level(candidates, router_config, calibrated_risk=calibrator.predict(raw_risk))
            else:
                router_config = RouterConfig(strategy=strategy, risk_target=error_risk_target)  # type: ignore[arg-type]
                chosen = select_level(candidates, router_config)
            pred, mask = per_level[chosen]  # type: ignore[index]
            accumulator.update(pred, mask)
            chosen_latencies.append(latency_by_level[chosen])  # type: ignore[index]
        result = accumulator.compute()
        strategy_results[strategy] = {
            "achieved_miou": result.miou,
            "avg_latency_ms": mean(chosen_latencies),
        }

    console.print(
        f"[{split_name}] fit n={len(fit_scores)} (mean error={mean(fit_errors):.4f}, "
        f"median={median(fit_errors):.4f}) "
        f"error_risk_target={error_risk_target:.4f} "
        f"({'explicit' if error_risk_target_override is not None else 'auto'}) "
        f"raw_risk_target={raw_risk_target:.4f} "
        f"({'explicit' if raw_risk_target_override is not None else 'auto'})"
    )
    return {
        "num_fit": len(fit_scores),
        "num_test": len(test_probe_risk),
        "error_risk_target": error_risk_target,
        "raw_risk_target": raw_risk_target,
        "calibrator": calibrator.to_dict(),
        "strategies": strategy_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    parser.add_argument("--lookup-table", type=Path, default=Path("outputs/benchmark_lookup_table.csv"))
    parser.add_argument("--eval-json", type=Path, default=None, help="evaluate_supernet.py --output-json, enables the oracle strategy")
    parser.add_argument("--device-id", required=True, help='deployment target device, e.g. "E1" or "E3"')
    parser.add_argument("--backend", required=True, help='deployment target backend, e.g. "hailo_hef" or "tensorrt_gpu"')
    parser.add_argument("--latency-field", default="end_to_end_p95_ms")
    parser.add_argument("--probe-level", default=None, help="default: the cheapest configured level")
    parser.add_argument(
        "--error-risk-target",
        type=float,
        default=None,
        help="override calibrated_risk's risk_target (error-probability scale, "
        "0-1) -- default: fit-half's mean observed error. A stricter (smaller) "
        "value forces more escalation to medium/large.",
    )
    parser.add_argument(
        "--raw-risk-target",
        type=float,
        default=None,
        help="override entropy's risk_target (raw softmax-entropy scale, nats, "
        "typically ~0.3-3) -- default: fit-half's mean raw score. Do NOT pass an "
        "error-probability-scale value here, or entropy over-escalates almost "
        "every image (see module docstring's 2026-09-15 scale-mismatch bug).",
    )
    parser.add_argument("--dataset", action="append", default=[], choices=["cityscapes", "acdc"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-calibrator", type=Path, default=None, help="save one split's fitted calibrator (last one evaluated) as JSON")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    console = Console()
    config = load_config(args.config)
    supernet = PaceSegSupernet(config.supernet).to(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=args.device)
    supernet.load_state_dict(checkpoint["model_state_dict"])
    supernet.eval()
    console.print(f"loaded checkpoint step={checkpoint['step']} config_hash={checkpoint['config_hash']}")

    probe_level: ElasticityLevel = args.probe_level or config.supernet.levels[0]
    cityscapes_root = _dataset_root(config, "cityscapes")
    acdc_root = _dataset_root(config, "acdc")

    with args.lookup_table.open() as f:
        lookup_rows = list(csv.DictReader(f))
    latency_by_level: dict[ElasticityLevel, float] = {}
    missing_levels = []
    for level in config.supernet.levels:
        matches = [
            r
            for r in lookup_rows
            if r["device_id"] == args.device_id and r["backend"] == args.backend and r["level"] == level
        ]
        if matches and matches[0].get(args.latency_field):
            latency_by_level[level] = float(matches[0][args.latency_field])
        else:
            latency_by_level[level] = _fallback_latency(level)
            missing_levels.append(level)
    if missing_levels:
        console.print(
            f"[yellow]warning: no {args.device_id}/{args.backend} lookup-table row for "
            f"level(s) {missing_levels} -- using synthetic placeholder latency, NOT a "
            f"real measurement, for those level(s)[/yellow]"
        )

    candidates = [
        ParetoPoint(
            level=level,
            device_id=args.device_id,
            backend=args.backend,
            precision="fp16",
            latency_ms=latency_by_level[level],
            miou=0.0,  # unused by static/entropy/calibrated_risk; see module docstring
            dataset="n/a",
        )
        for level in config.supernet.levels
    ]

    oracle_level: ElasticityLevel | None = None
    miou_by_level = json.loads(args.eval_json.read_text()) if args.eval_json else None

    datasets = args.dataset or ["cityscapes", "acdc"]
    split_names: list[str] = []
    if "cityscapes" in datasets and cityscapes_root is not None:
        split_names.append("cityscapes")
    if "acdc" in datasets and acdc_root is not None:
        split_names.extend(f"acdc/{c}" for c in ALL_CONDITIONS)

    results: dict[str, object] = {}
    for split_name in split_names:
        if miou_by_level is not None:
            oracle_points = build_pareto_points(
                [r for r in lookup_rows if r["device_id"] == args.device_id and r["backend"] == args.backend],
                miou_by_level,
                dataset=split_name,
                latency_field=args.latency_field,
            )
            oracle_level = max(oracle_points, key=lambda p: p.miou).level if oracle_points else None

        split_result = _evaluate_split(
            console,
            supernet,
            config,
            split_name,
            probe_level,
            candidates,
            latency_by_level,
            args.error_risk_target,
            args.raw_risk_target,
            args.device,
            cityscapes_root,
            acdc_root,
            oracle_level,
        )
        results[split_name] = split_result

        table = Table(title=f"router strategies -- {split_name} -- probe={probe_level} target={args.device_id}/{args.backend}")
        table.add_column("strategy")
        table.add_column("avg latency (ms)", justify="right")
        table.add_column("achieved mIoU", justify="right")
        strategy_results = cast(dict[str, dict[str, float]], split_result["strategies"])
        for strategy, values in strategy_results.items():
            table.add_row(strategy, f"{values['avg_latency_ms']:.3f}", f"{values['achieved_miou']:.4f}")
        console.print(table)

        if args.save_calibrator:
            calibrator_dict = cast(dict[str, object], split_result["calibrator"])
            RiskCalibrator.from_dict(calibrator_dict).save(args.save_calibrator)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        console.print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
