"""Router progressive ablation A->B->C->D (`docs/COORDINATION_LOG.md` open thread #2,
protocol locked with Codex 2026-09-21): wires `src/imavis_edge_seg/router/` into a
real inference pass and measures every cell against real data, on one deployment
target (`--device-id`/`--backend`) at a time.

    uv run python scripts/evaluate_router.py --checkpoint outputs/pace_seg_v1/checkpoints/step_00100000.pt --config configs/experiment/default.yaml --lookup-table outputs/benchmark_lookup_table.csv --device-id E3 --backend tensorrt_gpu --dataset cityscapes

Splits each dataset split's validation images in half by alternating index (fit-half:
even indices, held-out half: odd indices) -- `RESEARCH_PLAN.md` §5's "calibration is
fit on validation data only" applies here directly: fitting on the held-out half
would leak information the router isn't allowed to have. Unlike the original version
of this script, **both halves now run every elasticity level's forward pass**, not
just the probe level for the fit-half -- the per-level (cell C) calibrators need each
level's own fit-half observed error, and Codex's "cache prediction/error of every
(image, level) once" instruction means the rest of the sweep below is pure
post-processing over these cached (pred, mask) tensors, never re-inference.

Cells, per the locked protocol (see `router/policy.py`'s module docstring for the
exact decision rules):
- **A** (`calibrated_risk`), **B** (`latency_spacing_risk`), **C**
  (`candidate_specific_risk`): each swept across the SAME 5-point risk-target grid
  (`RISK_TARGET_QUANTILES` = Q10/25/50/75/90 of fit-half observed error, macro-averaged
  across splits so no split dominates -- `router.grid.macro_quantile_grid`), reporting
  a full quality-latency frontier, never a single manually-chosen threshold.
- **entropy**: its own 5-point grid, same quantiles but computed on the probe's raw
  (unbounded, nats-scale) risk score instead of error -- entropy and calibrated risk
  live on different scales, so their grids are computed separately even though both
  use the same target quantiles.
- **D** (`risk_latency_constrained`): the full method -- 4 (this device's own
  per-level p95 latencies, the only breakpoints where D's feasible candidate set can
  change) x 5 (the same risk-target grid) = 20 operating points.
- **static**: all 4 elasticity levels reported individually (free, no training/sweep
  needed, they anchor the Pareto frontier).
- **oracle**: budget-matched and per-image -- for each of the 4 device-latency
  budgets, per image, the candidate with the lowest *ground-truth* observed error
  among those within budget. No risk-target sweep (oracle doesn't use a risk score).

For every strategy that takes a risk-target grid (A/B/C/entropy), a single
budget-matched "operating point" is also picked per device-latency-budget, via
`router.grid.select_budget_matched_operating_point`'s pre-registered rule: among the
grid's points whose FIT-HALF mean latency doesn't exceed the budget, pick the
highest FIT-HALF quality (ties by lower latency); if none fits, fall back to the
cheapest point and mark it infeasible. This selection never looks at the held-out
half -- only the resulting held-out quality/latency/violation-rate is then reported,
for a fair per-budget comparison against D and the oracle.

Reported per split: the full risk-target frontier per strategy, the per-budget
table (A/B/C/D/entropy/oracle/static), `router.calibrator.prediction_inversion_rate`
(cell C diagnostic), and `router.selective_metrics.area_under_risk_coverage` for the
probe's own raw-risk-vs-observed-error signal (assesses the probe signal's quality
independent of any routing policy). mIoU-per-ms is deliberately never reported as a
headline number (`reports/router_v1_20260914.md`'s 2/7-mixed efficiency result is
why) -- use the quality-latency frontier and per-budget table instead.

**Scope note**: none of this measures the router's own runtime overhead (the probe
forward pass, calibrator lookup, decision logic) -- that is `docs/COORDINATION_LOG.md`
open thread #2's item 5, a separate hardware-overhead experiment, deferred until the
full method (D) shows a clear Pareto or equal-risk-latency advantage here first.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import cast

import torch
from rich.console import Console
from rich.table import Table

from imavis_edge_seg.config import ElasticityLevel, ExperimentConfig, RouterConfig, load_config
from imavis_edge_seg.data.acdc import ALL_CONDITIONS
from imavis_edge_seg.evaluation.data import build_acdc_eval_loader, build_cityscapes_eval_loader
from imavis_edge_seg.evaluation.metrics import ConfusionMatrixAccumulator
from imavis_edge_seg.models.supernet import PaceSegSupernet
from imavis_edge_seg.router.calibrator import fit_per_level_calibrators, prediction_inversion_rate
from imavis_edge_seg.router.grid import macro_quantile_grid, select_budget_matched_operating_point
from imavis_edge_seg.router.observed_error import compute_per_image_error
from imavis_edge_seg.router.policy import select_level
from imavis_edge_seg.router.risk_probe import compute_risk_score
from imavis_edge_seg.router.selective_metrics import area_under_risk_coverage, budget_violation_rate
from imavis_edge_seg.search.pareto import ParetoPoint
from imavis_edge_seg.training.checkpoint import load_checkpoint

_PredMask = tuple[torch.Tensor, torch.Tensor]


@dataclass
class _SplitData:
    fit_raw_scores: list[float]
    fit_pred_mask: dict[ElasticityLevel, list[_PredMask]]
    test_raw_scores: list[float]
    test_pred_mask: dict[ElasticityLevel, list[_PredMask]]


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
def _collect_split_data(
    supernet: torch.nn.Module,
    config: ExperimentConfig,
    split_name: str,
    probe_level: ElasticityLevel,
    device: str,
    cityscapes_root: Path | None,
    acdc_root: Path | None,
) -> _SplitData:
    """Runs every configured elasticity level's forward pass on every image, once,
    caching (pred, mask) CPU tensors -- both halves get every level (not just the
    fit-half's probe level, unlike the original version of this script), since cell
    C's per-level calibrators need each level's own fit-half observed error. Every
    later cell/grid computation in `main()` is pure post-processing over this cache,
    never a repeated forward pass."""
    loaders = _build_loaders(config, split_name, cityscapes_root, acdc_root)
    iterators = {level: iter(loader) for level, loader in loaders.items()}
    num_images = len(loaders[probe_level].dataset)  # type: ignore[arg-type]

    fit_raw_scores: list[float] = []
    fit_pred_mask: dict[ElasticityLevel, list[_PredMask]] = {level: [] for level in loaders}
    test_raw_scores: list[float] = []
    test_pred_mask: dict[ElasticityLevel, list[_PredMask]] = {level: [] for level in loaders}

    for i in range(num_images):
        batches = {level: next(iterators[level]) for level in loaders}
        is_fit = i % 2 == 0

        probe_image, probe_mask = batches[probe_level]
        probe_image, probe_mask = probe_image.to(device), probe_mask.to(device)
        probe_logits = supernet(probe_image, probe_level)

        # Fit-half: risk score computed WITH ground truth (matches the error it's
        # being calibrated against -- same pixels excluded on both sides).
        # Held-out: risk score computed WITHOUT ground truth, matching real inference.
        raw_score = float(compute_risk_score(probe_logits, probe_mask if is_fit else None))
        (fit_raw_scores if is_fit else test_raw_scores).append(raw_score)

        target = fit_pred_mask if is_fit else test_pred_mask
        for level, (image, mask) in batches.items():
            image, mask = image.to(device), mask.to(device)
            logits = probe_logits if level == probe_level else supernet(image, level)
            pred = logits.argmax(dim=1)
            target[level].append((pred.cpu(), mask.cpu()))

    if not fit_raw_scores or not test_raw_scores:
        raise ValueError(f"{split_name}: need at least 2 val images (got {num_images}) to split fit/test")
    return _SplitData(fit_raw_scores, fit_pred_mask, test_raw_scores, test_pred_mask)


def _per_image_errors(pred_mask: list[_PredMask]) -> list[float]:
    return [float(compute_per_image_error(pred, mask)) for pred, mask in pred_mask]


def _accumulate_strategy(
    candidates: list[ParetoPoint],
    config: RouterConfig,
    raw_scores: list[float] | None,
    calibrated_risks: list[float] | None,
    per_level_risks: list[dict[ElasticityLevel, float]] | None,
    pred_mask: dict[ElasticityLevel, list[_PredMask]],
    latency_by_level: dict[ElasticityLevel, float],
) -> tuple[float, float, list[float], Counter[ElasticityLevel]]:
    """Runs `select_level` once per image under one fixed `RouterConfig`, accumulating
    achieved mIoU (via a real confusion matrix over the chosen candidate's cached
    prediction), mean chosen latency, the list of per-image chosen latencies (for
    `budget_violation_rate`), and the routing distribution. `raw_scores`/
    `calibrated_risks`/`per_level_risks` are parallel-length lists (one entry per
    image); pass only the one(s) `config.strategy` actually needs."""
    n = len(next(iter(pred_mask.values())))
    accumulator = ConfusionMatrixAccumulator()
    chosen_latencies: list[float] = []
    distribution: Counter[ElasticityLevel] = Counter()
    for i in range(n):
        chosen = select_level(
            candidates,
            config,
            calibrated_risk=calibrated_risks[i] if calibrated_risks is not None else None,
            raw_risk=raw_scores[i] if raw_scores is not None else None,
            per_level_risk=per_level_risks[i] if per_level_risks is not None else None,
        )
        pred, mask = pred_mask[chosen][i]
        accumulator.update(pred, mask)
        chosen_latencies.append(latency_by_level[chosen])
        distribution[chosen] += 1
    result = accumulator.compute()
    return result.miou, mean(chosen_latencies), chosen_latencies, distribution


def _oracle(
    budget_ms: float,
    ordered_levels: list[ElasticityLevel],
    latency_by_level: dict[ElasticityLevel, float],
    errors: dict[ElasticityLevel, list[float]],
    pred_mask: dict[ElasticityLevel, list[_PredMask]],
) -> tuple[float, float, list[float]]:
    """Budget-matched, per-image oracle (locked with Codex, replaces the old
    fixed-per-split "best aggregate mIoU" pick): for each image, the candidate with
    the lowest *ground-truth* observed error among those within `budget_ms`. Uses the
    same cached (pred, mask) tensors as every other cell -- no re-inference, and no
    external `--eval-json` needed any more."""
    in_budget = [level for level in ordered_levels if latency_by_level[level] <= budget_ms]
    if not in_budget:
        in_budget = [ordered_levels[0]]  # cheapest overall, mirrors cell D's rule 4
    n = len(next(iter(pred_mask.values())))
    accumulator = ConfusionMatrixAccumulator()
    chosen_latencies: list[float] = []
    for i in range(n):
        chosen = min(in_budget, key=lambda level: errors[level][i])
        pred, mask = pred_mask[chosen][i]
        accumulator.update(pred, mask)
        chosen_latencies.append(latency_by_level[chosen])
    result = accumulator.compute()
    return result.miou, mean(chosen_latencies), chosen_latencies


def _evaluate_split(
    console: Console,
    split_name: str,
    probe_level: ElasticityLevel,
    ordered_levels: list[ElasticityLevel],
    candidates: list[ParetoPoint],
    latency_by_level: dict[ElasticityLevel, float],
    device_budget_grid: list[float],
    risk_target_grid: list[float],
    entropy_threshold_grid: list[float],
    data: _SplitData,
) -> dict[str, object]:
    fit_errors = {level: _per_image_errors(pm) for level, pm in data.fit_pred_mask.items()}
    test_errors = {level: _per_image_errors(pm) for level, pm in data.test_pred_mask.items()}

    per_level_calibrators = fit_per_level_calibrators(data.fit_raw_scores, fit_errors)
    probe_calibrator = per_level_calibrators[probe_level]  # cell A/B's single scalar risk

    fit_calibrated = [probe_calibrator.predict(s) for s in data.fit_raw_scores]
    test_calibrated = [probe_calibrator.predict(s) for s in data.test_raw_scores]
    fit_per_level_risk = [
        {level: cal.predict(s) for level, cal in per_level_calibrators.items()} for s in data.fit_raw_scores
    ]
    test_per_level_risk = [
        {level: cal.predict(s) for level, cal in per_level_calibrators.items()} for s in data.test_raw_scores
    ]

    def _frontier(strategy: str, grid: list[float], use_raw: bool, use_per_level: bool) -> list[dict[str, object]]:
        points = []
        for target in grid:
            config = RouterConfig(strategy=strategy, risk_target=target)  # type: ignore[arg-type]
            fit_miou, fit_latency, _, _ = _accumulate_strategy(
                candidates,
                config,
                data.fit_raw_scores if use_raw else None,
                fit_calibrated if not use_raw and not use_per_level else None,
                fit_per_level_risk if use_per_level else None,
                data.fit_pred_mask,
                latency_by_level,
            )
            test_miou, test_latency, test_lat_list, distribution = _accumulate_strategy(
                candidates,
                config,
                data.test_raw_scores if use_raw else None,
                test_calibrated if not use_raw and not use_per_level else None,
                test_per_level_risk if use_per_level else None,
                data.test_pred_mask,
                latency_by_level,
            )
            points.append(
                {
                    "risk_target": target,
                    "fit_half": {"achieved_miou": fit_miou, "avg_latency_ms": fit_latency},
                    "held_out": {
                        "achieved_miou": test_miou,
                        "avg_latency_ms": test_latency,
                        "chosen_latencies_ms": test_lat_list,
                        "routing_distribution": dict(distribution),
                    },
                }
            )
        return points

    frontiers = {
        "calibrated_risk": _frontier("calibrated_risk", risk_target_grid, use_raw=False, use_per_level=False),
        "latency_spacing_risk": _frontier(
            "latency_spacing_risk", risk_target_grid, use_raw=False, use_per_level=False
        ),
        "candidate_specific_risk": _frontier(
            "candidate_specific_risk", risk_target_grid, use_raw=False, use_per_level=True
        ),
        "entropy": _frontier("entropy", entropy_threshold_grid, use_raw=True, use_per_level=False),
    }

    # Cell D: 4 device budgets x 5 risk targets = 20 operating points, evaluated
    # directly on held-out (D's own policy already enforces the budget internally).
    d_grid: list[dict[str, object]] = []
    for budget in device_budget_grid:
        for target in risk_target_grid:
            config = RouterConfig(strategy="risk_latency_constrained", risk_target=target, latency_budget_ms=budget)
            miou, avg_latency, lat_list, distribution = _accumulate_strategy(
                candidates, config, None, None, test_per_level_risk, data.test_pred_mask, latency_by_level
            )
            d_grid.append(
                {
                    "latency_budget_ms": budget,
                    "risk_target": target,
                    "achieved_miou": miou,
                    "avg_latency_ms": avg_latency,
                    "violation_rate": budget_violation_rate(lat_list, budget),
                    "routing_distribution": dict(distribution),
                }
            )

    # Per-budget table: A/B/C/entropy each get ONE representative point per budget,
    # chosen on fit-half only (router.grid.select_budget_matched_operating_point);
    # D's 20-point grid is reduced to one per budget the same way, among its 5
    # risk-target sub-points at that budget; static/oracle need no selection.
    at_budget: dict[str, dict[float, object]] = {name: {} for name in (*frontiers, "risk_latency_constrained")}
    for strategy_name, points in frontiers.items():
        for budget in device_budget_grid:
            operating_points = [
                (cast(dict[str, float], p["fit_half"])["avg_latency_ms"], cast(dict[str, float], p["fit_half"])["achieved_miou"], p)
                for p in points
            ]
            chosen, feasible = select_budget_matched_operating_point(operating_points, budget)
            chosen_point = cast(dict[str, object], chosen)
            held_out = cast(dict[str, object], chosen_point["held_out"])
            at_budget[strategy_name][budget] = {
                "risk_target": chosen_point["risk_target"],
                "feasible": feasible,
                "achieved_miou": held_out["achieved_miou"],
                "avg_latency_ms": held_out["avg_latency_ms"],
                "violation_rate": budget_violation_rate(cast(list[float], held_out["chosen_latencies_ms"]), budget),
            }
    for budget in device_budget_grid:
        sub_points = [p for p in d_grid if p["latency_budget_ms"] == budget]
        operating_points = [(cast(float, p["avg_latency_ms"]), cast(float, p["achieved_miou"]), p) for p in sub_points]
        chosen, feasible = select_budget_matched_operating_point(operating_points, budget)
        chosen_point = cast(dict[str, object], chosen)
        at_budget["risk_latency_constrained"][budget] = {
            "risk_target": chosen_point["risk_target"],
            "feasible": feasible,
            "achieved_miou": chosen_point["achieved_miou"],
            "avg_latency_ms": chosen_point["avg_latency_ms"],
            "violation_rate": chosen_point["violation_rate"],
        }

    oracle_by_budget: dict[float, object] = {}
    for budget in device_budget_grid:
        miou, avg_latency, lat_list = _oracle(budget, ordered_levels, latency_by_level, test_errors, data.test_pred_mask)
        oracle_by_budget[budget] = {
            "achieved_miou": miou,
            "avg_latency_ms": avg_latency,
            "violation_rate": budget_violation_rate(lat_list, budget),
        }

    static_points: dict[ElasticityLevel, dict[str, object]] = {}
    for level in ordered_levels:
        accumulator = ConfusionMatrixAccumulator()
        for pred, mask in data.test_pred_mask[level]:
            accumulator.update(pred, mask)
        static_points[level] = {"latency_ms": latency_by_level[level], "achieved_miou": accumulator.compute().miou}

    inversion_rate = prediction_inversion_rate(ordered_levels, test_per_level_risk)
    probe_signal_aurc = area_under_risk_coverage(data.test_raw_scores, test_errors[probe_level])

    console.print(
        f"[{split_name}] fit n={len(data.fit_raw_scores)} test n={len(data.test_raw_scores)} "
        f"risk_target_grid={[round(v, 4) for v in risk_target_grid]} "
        f"entropy_threshold_grid={[round(v, 4) for v in entropy_threshold_grid]} "
        f"device_budget_grid={[round(v, 3) for v in device_budget_grid]} "
        f"prediction_inversion_rate={inversion_rate:.4f} probe_signal_aurc={probe_signal_aurc:.4f}"
    )
    return {
        "risk_target_grid": risk_target_grid,
        "entropy_threshold_grid": entropy_threshold_grid,
        "device_budget_grid": device_budget_grid,
        "prediction_inversion_rate": inversion_rate,
        "probe_signal_aurc": probe_signal_aurc,
        "static": static_points,
        "frontier": frontiers,
        "risk_latency_constrained_grid": d_grid,
        "at_budget": at_budget,
        "oracle": oracle_by_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    parser.add_argument("--lookup-table", type=Path, default=Path("outputs/benchmark_lookup_table.csv"))
    parser.add_argument("--device-id", required=True, help='deployment target device, e.g. "E1" or "E3"')
    parser.add_argument("--backend", required=True, help='deployment target backend, e.g. "hailo_hef" or "tensorrt_gpu"')
    parser.add_argument("--latency-field", default="end_to_end_p95_ms")
    parser.add_argument("--probe-level", default=None, help="default: the cheapest configured level")
    parser.add_argument("--dataset", action="append", default=[], choices=["cityscapes", "acdc"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    ordered_levels: list[ElasticityLevel] = list(config.supernet.levels)
    cityscapes_root = _dataset_root(config, "cityscapes")
    acdc_root = _dataset_root(config, "acdc")

    with args.lookup_table.open() as f:
        lookup_rows = list(csv.DictReader(f))
    latency_by_level: dict[ElasticityLevel, float] = {}
    missing_levels = []
    for level in ordered_levels:
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
    # Cell D/oracle's primary budget grid: the device's own 4 per-level latencies --
    # the only breakpoints where the feasible candidate set can change (Codex,
    # 2026-09-21) -- deduplicated/sorted, not log-spaced.
    device_budget_grid = sorted({latency_by_level[level] for level in ordered_levels})

    candidates = [
        ParetoPoint(
            level=level,
            device_id=args.device_id,
            backend=args.backend,
            precision="fp16",
            latency_ms=latency_by_level[level],
            miou=0.0,  # unused by any strategy here; achieved mIoU is measured directly
            dataset="n/a",
        )
        for level in ordered_levels
    ]

    datasets = args.dataset or ["cityscapes", "acdc"]
    split_names: list[str] = []
    if "cityscapes" in datasets and cityscapes_root is not None:
        split_names.append("cityscapes")
    if "acdc" in datasets and acdc_root is not None:
        split_names.extend(f"acdc/{c}" for c in ALL_CONDITIONS)

    # Pass 1: collect every split's cached predictions first -- the risk-target and
    # entropy-threshold grids are macro-averaged ACROSS splits (Codex: "một bộ tau
    # chung cho E1/E2/E3/E5; không tính lại theo device" -- device-independent, and
    # by the same logic here, computed once from all splits' fit-halves, not
    # per-split), so every split's fit-half data must exist before either grid can be
    # locked.
    split_data: dict[str, _SplitData] = {}
    for split_name in split_names:
        console.print(f"collecting predictions for {split_name}...")
        split_data[split_name] = _collect_split_data(
            supernet, config, split_name, probe_level, args.device, cityscapes_root, acdc_root
        )

    error_pools = []
    raw_pools = []
    for data in split_data.values():
        errors_this_split = [
            e for level in ordered_levels for e in _per_image_errors(data.fit_pred_mask[level])
        ]
        error_pools.append(errors_this_split)
        raw_pools.append(data.fit_raw_scores)
    risk_target_grid = macro_quantile_grid(error_pools)
    entropy_threshold_grid = macro_quantile_grid(raw_pools)
    console.print(
        f"locked grids (macro-averaged across {len(split_names)} splits' fit-halves): "
        f"risk_target_grid={[round(v, 4) for v in risk_target_grid]} "
        f"entropy_threshold_grid={[round(v, 4) for v in entropy_threshold_grid]}"
    )

    # Pass 2: evaluate held-out per split, using the now-locked grids.
    results: dict[str, object] = {}
    for split_name in split_names:
        split_result = _evaluate_split(
            console,
            split_name,
            probe_level,
            ordered_levels,
            candidates,
            latency_by_level,
            device_budget_grid,
            risk_target_grid,
            entropy_threshold_grid,
            split_data[split_name],
        )
        results[split_name] = split_result

        table = Table(title=f"router at-budget summary -- {split_name} -- probe={probe_level} target={args.device_id}/{args.backend}")
        table.add_column("strategy")
        table.add_column("budget (ms)", justify="right")
        table.add_column("avg latency (ms)", justify="right")
        table.add_column("achieved mIoU", justify="right")
        table.add_column("violation rate", justify="right")
        table.add_column("feasible")
        at_budget = cast(dict[str, dict[float, dict[str, object]]], split_result["at_budget"])
        for strategy_name, by_budget in at_budget.items():
            for budget, point in by_budget.items():
                table.add_row(
                    strategy_name,
                    f"{budget:.3f}",
                    f"{cast(float, point['avg_latency_ms']):.3f}",
                    f"{cast(float, point['achieved_miou']):.4f}",
                    f"{cast(float, point['violation_rate']):.3f}",
                    str(point["feasible"]),
                )
        oracle_by_budget = cast(dict[float, dict[str, object]], split_result["oracle"])
        for budget, point in oracle_by_budget.items():
            table.add_row(
                "oracle",
                f"{budget:.3f}",
                f"{cast(float, point['avg_latency_ms']):.3f}",
                f"{cast(float, point['achieved_miou']):.4f}",
                f"{cast(float, point['violation_rate']):.3f}",
                "True",
            )
        static_points = cast(dict[ElasticityLevel, dict[str, object]], split_result["static"])
        for level, point in static_points.items():
            table.add_row(f"static_{level}", "n/a", f"{cast(float, point['latency_ms']):.3f}", f"{cast(float, point['achieved_miou']):.4f}", "n/a", "n/a")
        console.print(table)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, default=str))
        console.print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
