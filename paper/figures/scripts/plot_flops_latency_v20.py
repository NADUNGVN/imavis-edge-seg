"""V20 RQ1 figure: the V17 single-panel layout on the landscape rerun.

Inputs (landscape, 2026-10-04):
  reports/landscape_20261004/flops_landscape.json
  outputs/benchmark_lookup_table_landscape.csv   (aggregate rows, run_index == -1)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pace_style_v13 import save_all
from plot_flops_latency_v13 import LEVELS
from plot_flops_latency_v17 import DEVICES, build_figure


def load_data(repo_root: Path) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    raw = json.loads((repo_root / "reports/landscape_20261004/flops_landscape.json").read_text())["supernet"]
    gflops = {lv: float(raw[lv]) / 1e9 for lv in LEVELS}
    latency: dict[str, dict[str, float]] = {}
    with (repo_root / "outputs/benchmark_lookup_table_landscape.csv").open(encoding="utf-8", newline="") as h:
        for row in csv.DictReader(h):
            if row["run_index"] == "-1" and row["device_id"] in DEVICES and row["level"] in LEVELS:
                latency.setdefault(row["device_id"], {})[row["level"]] = float(row["end_to_end_mean_ms"])
    if set(latency) != set(DEVICES) or any(set(v) != set(LEVELS) for v in latency.values()):
        raise RuntimeError("landscape LUT must contain all four levels for E1, E2, E3, E5")
    return gflops, latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument("--output-stem", type=Path,
                        default=Path(__file__).parents[1] / "generated" / "fig5_flops_latency_v20")
    args = parser.parse_args()
    gflops, latency = load_data(args.repo_root.resolve())
    save_all(build_figure(gflops, latency), args.output_stem)
    for d in DEVICES:
        print(d, {lv: round(latency[d][lv], 2) for lv in LEVELS}, f"{latency[d]['large'] / latency[d]['tiny']:.1f}x")
    print("flops ratio", round(gflops["large"] / gflops["tiny"], 1))


if __name__ == "__main__":
    main()
