"""Compute FLOPs per elasticity level for the supernet (`docs/RESEARCH_PLAN.md` §7's
FLOPs-aware baseline axis / RQ1). Untrained weights only -- FLOPs depend only on the
graph's shapes, not weight values (same convention `export_all_levels.py` uses for
compiler smoke tests).

    uv run python scripts/compute_flops.py --output-json outputs/flops_by_level.json

Add --baseline <name> (repeatable) to also report a required baseline's FLOPs at the
same resolution as the supernet's largest level, for reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from imavis_edge_seg.config import ExperimentConfig
from imavis_edge_seg.models import PaceSegSupernet
from imavis_edge_seg.models.baselines import BASELINE_NAMES, build_baseline_model
from imavis_edge_seg.search.flops import count_flops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="flops")
    parser.add_argument(
        "--baseline", action="append", default=[], choices=BASELINE_NAMES,
        help="also report this required baseline's FLOPs (repeatable)",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    console = Console()
    config = ExperimentConfig(experiment_id=args.experiment_id)
    supernet = PaceSegSupernet(config.supernet)
    supernet.eval()

    table = Table(title="Supernet FLOPs by elasticity level")
    table.add_column("level")
    table.add_column("resolution")
    table.add_column("params")
    table.add_column("GFLOPs", justify="right")

    flops_by_level: dict[str, int] = {}
    for level in config.supernet.levels:
        height, width = config.supernet.input_resolutions[level]
        x = torch.randn(1, 3, height, width)

        def _run_supernet(x: torch.Tensor = x, level: str = level) -> torch.Tensor:
            return supernet(x, level)  # type: ignore[no-any-return]

        flops = count_flops(supernet, _run_supernet)
        flops_by_level[level] = flops
        n_params = sum(p.numel() for p in supernet.parameters())
        table.add_row(level, f"{width}x{height}", f"{n_params:,}", f"{flops / 1e9:.4f}")

    console.print(table)

    results: dict[str, object] = {"supernet": flops_by_level}

    if args.baseline:
        largest = config.supernet.levels[-1]
        height, width = config.supernet.input_resolutions[largest]
        baseline_table = Table(title=f"Baseline FLOPs at {width}x{height} (supernet's largest level's resolution)")
        baseline_table.add_column("model")
        baseline_table.add_column("params")
        baseline_table.add_column("GFLOPs", justify="right")
        baseline_flops: dict[str, int] = {}
        for name in args.baseline:
            model = build_baseline_model(name, num_classes=config.supernet.num_classes)
            model.eval()
            x = torch.randn(1, 3, height, width)

            def _run_baseline(x: torch.Tensor = x, model: torch.nn.Module = model) -> torch.Tensor:
                return model(x)  # type: ignore[no-any-return]

            flops = count_flops(model, _run_baseline)
            baseline_flops[name] = flops
            n_params = sum(p.numel() for p in model.parameters())
            baseline_table.add_row(name, f"{n_params:,}", f"{flops / 1e9:.4f}")
        console.print(baseline_table)
        results["baselines"] = baseline_flops

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        console.print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
