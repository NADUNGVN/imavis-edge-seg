"""RQ1 budget sweep (proposed by Codex, 2026-09-20): stronger evidence than the
single 10ms-budget table in `scripts/flops_vs_latency_baseline.py` -- sweeps many
latency budgets x every (reference device that calibrates the FLOPs proxy) x every
target device, reporting mis-selection rate, accuracy regret, and unused-budget
slack/violation rate. No training needed -- reuses the existing real LUT + eval JSON.

    uv run python scripts/rq1_budget_sweep.py --flops-json outputs/flops_by_level.json --lookup-table outputs/benchmark_lookup_table.csv --eval-json reports/eval_pace_seg_v1_aug_seed0_step100000.json --dataset cityscapes --output-json reports/rq1_budget_sweep_cityscapes.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from imavis_edge_seg.config import ElasticityLevel
from imavis_edge_seg.search.flops import evaluate_flops_proxy_at_budget, fit_flops_to_latency_rate
from imavis_edge_seg.search.pareto import build_pareto_points


def _log_spaced_budgets(min_ms: float, max_ms: float, num: int) -> list[float]:
    if min_ms <= 0:
        min_ms = 0.01
    ratio = (max_ms / min_ms) ** (1 / (num - 1))
    return [min_ms * (ratio**i) for i in range(num)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flops-json", type=Path, required=True)
    parser.add_argument("--lookup-table", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--latency-field", default="end_to_end_p95_ms")
    parser.add_argument("--num-budgets", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    console = Console()
    flops_by_level: dict[ElasticityLevel, int] = json.loads(args.flops_json.read_text())["supernet"]
    miou_by_level = json.loads(args.eval_json.read_text())
    with args.lookup_table.open() as f:
        lookup_rows = list(csv.DictReader(f))

    targets = sorted({(r["device_id"], r["backend"]) for r in lookup_rows})
    device_ids = sorted({device_id for device_id, _ in targets})

    all_latencies = [float(r[args.latency_field]) for r in lookup_rows]
    budgets = _log_spaced_budgets(min(all_latencies) * 0.8, max(all_latencies) * 1.2, args.num_budgets)

    # Real Pareto points and a per-device latency-by-level dict, reused for both the
    # reference-rate fit and the per-budget evaluation below.
    real_points_by_target = {}
    latency_by_level_by_device: dict[str, dict[ElasticityLevel, float]] = {}
    for device_id, backend in targets:
        rows = [r for r in lookup_rows if r["device_id"] == device_id and r["backend"] == backend]
        real_points_by_target[(device_id, backend)] = build_pareto_points(
            rows, miou_by_level, dataset=args.dataset, latency_field=args.latency_field
        )
        latency_by_level_by_device[device_id] = {
            r["level"]: float(r[args.latency_field]) for r in rows  # type: ignore[misc]
        }

    summary_table = Table(title=f"RQ1 budget sweep -- {args.dataset}, {len(budgets)} budgets ({min(budgets):.2f}-{max(budgets):.2f}ms)")
    summary_table.add_column("reference device")
    summary_table.add_column("target device")
    summary_table.add_column("mis-selection %", justify="right")
    summary_table.add_column("mean accuracy regret", justify="right")
    summary_table.add_column("budget-violation %", justify="right")
    summary_table.add_column("mean unused slack (ms)", justify="right")

    by_reference: dict[str, dict[str, dict[str, float]]] = {}
    results: dict[str, object] = {"dataset": args.dataset, "budgets_ms": budgets, "by_reference": by_reference}

    for reference_device in device_ids:
        rate = fit_flops_to_latency_rate(flops_by_level, latency_by_level_by_device[reference_device])
        predicted_latency = {level: flops * rate for level, flops in flops_by_level.items()}

        for device_id, backend in targets:
            real_points = real_points_by_target[(device_id, backend)]
            outcomes = [
                evaluate_flops_proxy_at_budget(real_points, predicted_latency, b) for b in budgets
            ]
            n = len(outcomes)
            mis_rate = 100.0 * sum(o.mis_selected for o in outcomes) / n
            regrets = [o.accuracy_regret for o in outcomes if o.accuracy_regret is not None]
            mean_regret = sum(regrets) / len(regrets) if regrets else float("nan")
            slacks = [o.slack_ms for o in outcomes if o.slack_ms is not None]
            violation_rate = 100.0 * sum(1 for s in slacks if s < 0) / len(slacks) if slacks else float("nan")
            mean_slack = sum(slacks) / len(slacks) if slacks else float("nan")

            summary_table.add_row(
                reference_device, f"{device_id}/{backend}",
                f"{mis_rate:.1f}", f"{mean_regret:+.4f}", f"{violation_rate:.1f}", f"{mean_slack:+.2f}",
            )
            by_reference.setdefault(reference_device, {})[f"{device_id}/{backend}"] = {
                "mis_selection_pct": mis_rate,
                "mean_accuracy_regret": mean_regret,
                "budget_violation_pct": violation_rate,
                "mean_unused_slack_ms": mean_slack,
            }

    console.print(summary_table)
    console.print(
        "[dim]mis-selection %: fraction of swept budgets where the FLOPs proxy picks a "
        "different level than real measured latency would. accuracy regret: real mIoU "
        "minus the proxy's REAL mIoU (positive = proxy underperforms). budget-violation "
        "%: fraction of budgets where the proxy's chosen level's REAL latency actually "
        "exceeds the budget (negative slack). mean unused slack: average budget left over "
        "(negative = average violation).[/dim]"
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        console.print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
