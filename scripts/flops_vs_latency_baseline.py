"""Quantifies RQ1's actual hypothesis: does a FLOPs-aware selection baseline pick a
worse level than measured-latency-aware selection would, on a real device? (`docs/RESEARCH_PLAN.md`
§7's "FLOPs-aware vs latency-aware... selection" baseline axis.)

Calibrates a simple FLOPs -> latency proxy on one *reference* device (a linear fit
through the origin, latency = k * FLOPs, fit by least squares across all 4 levels),
then uses that SAME proxy -- as a FLOPs-only selection method would have to, since it
has no per-device measurement -- to predict latency and pick a level under a fixed
budget on every *other* device. Compares against what that device's own real measured
latency would have selected (`search.pareto.select_under_latency_budget`).

    uv run python scripts/flops_vs_latency_baseline.py --flops-json outputs/flops_by_level.json --lookup-table outputs/benchmark_lookup_table.csv --eval-json reports/eval_pace_seg_v1_aug_seed0_step100000.json --dataset cityscapes --reference-device E3 --reference-backend tensorrt_gpu --budget-ms 10
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from rich.console import Console
from rich.table import Table

from imavis_edge_seg.config import ElasticityLevel
from imavis_edge_seg.search.flops import fit_flops_to_latency_rate
from imavis_edge_seg.search.pareto import build_pareto_points, select_under_latency_budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flops-json", type=Path, required=True, help="scripts/compute_flops.py --output-json")
    parser.add_argument("--lookup-table", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True, help="evaluate_supernet.py --output-json")
    parser.add_argument("--dataset", required=True, help='e.g. "cityscapes"')
    parser.add_argument("--reference-device", required=True, help="device to calibrate the FLOPs->latency rate from, e.g. E3")
    parser.add_argument("--reference-backend", required=True)
    parser.add_argument("--latency-field", default="end_to_end_p95_ms")
    parser.add_argument("--budget-ms", type=float, default=10.0)
    args = parser.parse_args()

    console = Console()
    flops_data = json.loads(args.flops_json.read_text())
    flops_by_level: dict[ElasticityLevel, int] = flops_data["supernet"]
    miou_by_level = json.loads(args.eval_json.read_text())
    with args.lookup_table.open() as f:
        lookup_rows = list(csv.DictReader(f))

    all_targets = sorted({(r["device_id"], r["backend"]) for r in lookup_rows})
    reference_rows = [
        r for r in lookup_rows if r["device_id"] == args.reference_device and r["backend"] == args.reference_backend
    ]
    reference_latency: dict[ElasticityLevel, float] = {
        r["level"]: float(r[args.latency_field]) for r in reference_rows  # type: ignore[misc]
    }
    rate = fit_flops_to_latency_rate(flops_by_level, reference_latency)
    console.print(
        f"Calibrated FLOPs->latency rate from {args.reference_device}/{args.reference_backend}: "
        f"{rate:.6f} ms/FLOP ({rate * 1e9:.4f} ms/GFLOP)"
    )

    predicted_latency = {level: flops * rate for level, flops in flops_by_level.items()}

    table = Table(title=f"FLOPs-proxy (calibrated on {args.reference_device}) vs. real measured latency -- {args.dataset}")
    table.add_column("device/backend")
    table.add_column("real pick (budget)", justify="center")
    table.add_column("FLOPs-proxy pick (same budget)", justify="center")
    table.add_column("match?", justify="center")
    table.add_column("worst per-level prediction error", justify="right")

    for device_id, backend in all_targets:
        device_rows = [r for r in lookup_rows if r["device_id"] == device_id and r["backend"] == backend]
        real_points = build_pareto_points(device_rows, miou_by_level, dataset=args.dataset, latency_field=args.latency_field)
        real_pick = select_under_latency_budget(real_points, args.budget_ms)

        # Same points, but with the FLOPs-proxy's *predicted* latency swapped in for the
        # real one, so the identical budget-selection function applies fairly to both.
        proxy_points = [replace(p, latency_ms=predicted_latency[p.level]) for p in real_points]
        proxy_pick = select_under_latency_budget(proxy_points, args.budget_ms)

        real_latency_by_level = {p.level: p.latency_ms for p in real_points}
        errors = [
            abs(predicted_latency[level] - real_latency_by_level[level]) / real_latency_by_level[level]
            for level in real_latency_by_level
            if level in predicted_latency and real_latency_by_level[level] > 0
        ]
        worst_error = max(errors) * 100 if errors else float("nan")

        match = "same" if (real_pick and proxy_pick and real_pick.level == proxy_pick.level) else "DIFFERENT"
        table.add_row(
            f"{device_id}/{backend}",
            real_pick.level if real_pick else "(none fit)",
            proxy_pick.level if proxy_pick else "(none fit)",
            match,
            f"{worst_error:.0f}%",
        )

    console.print(table)
    console.print(
        "[dim]'FLOPs-proxy pick' uses ONLY the reference device's calibrated rate -- "
        "exactly what a FLOPs-only selection method (no per-device measurement) would "
        "have to do.[/dim]"
    )


if __name__ == "__main__":
    main()
