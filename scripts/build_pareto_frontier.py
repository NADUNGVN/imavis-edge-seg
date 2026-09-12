"""Build and print the Pareto latency/mIoU frontier per deployment target
(`docs/RESEARCH_PLAN.md` Contribution 2), from a benchmark lookup table CSV
(`scripts/build_benchmark_lookup_table.py`) and an evaluation JSON
(`scripts/evaluate_supernet.py --output-json`).

    uv run python scripts/build_pareto_frontier.py --lookup-table outputs/benchmark_lookup_table.csv --eval-json reports/eval_pace_seg_v1_step100000.json --dataset cityscapes
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from imavis_edge_seg.search.pareto import (
    build_pareto_points,
    pareto_frontiers_per_target,
    select_under_latency_budget,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookup-table", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help='e.g. "cityscapes" or "acdc/fog"')
    parser.add_argument(
        "--latency-field",
        default="end_to_end_p95_ms",
        help="lookup-table column to use as the cost axis (default: p95, tail latency)",
    )
    parser.add_argument(
        "--budget-ms", type=float, default=None, help="if given, also print the best level under this budget per target"
    )
    args = parser.parse_args()

    console = Console()
    with args.lookup_table.open() as f:
        rows = list(csv.DictReader(f))
    miou_by_level = json.loads(args.eval_json.read_text())

    points = build_pareto_points(rows, miou_by_level, dataset=args.dataset, latency_field=args.latency_field)
    if not points:
        console.print(f"[yellow]no points -- no lookup-table row matched a mIoU entry for dataset {args.dataset!r}[/yellow]")
        return

    frontiers = pareto_frontiers_per_target(points)
    for (device_id, backend), frontier in sorted(frontiers.items()):
        table = Table(title=f"Pareto frontier -- {device_id} / {backend} / {args.dataset}")
        table.add_column("level")
        table.add_column(args.latency_field, justify="right")
        table.add_column("mIoU", justify="right")
        for point in frontier:
            table.add_row(point.level, f"{point.latency_ms:.3f}", f"{point.miou:.4f}")
        console.print(table)

        if args.budget_ms is not None:
            best = select_under_latency_budget(frontier, args.budget_ms)
            console.print(
                f"  best under {args.budget_ms}ms budget: "
                f"{best.level if best else '(none fit)'}"
            )


if __name__ == "__main__":
    main()
