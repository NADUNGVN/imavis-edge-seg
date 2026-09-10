"""Merge every `BenchmarkRecord` JSON under a directory into one latency/energy lookup
table CSV, per `docs/RESEARCH_PLAN.md` §5.3 B / §5.4.

    uv run python scripts/build_benchmark_lookup_table.py --records-dir outputs/benchmark_records --output outputs/benchmark_lookup_table.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from imavis_edge_seg.benchmark.lookup_table import build_lookup_table, write_lookup_table_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    console = Console()
    rows = build_lookup_table(args.records_dir)
    if not rows:
        console.print(f"[yellow]no BenchmarkRecord *.json files found under {args.records_dir}[/yellow]")
        return

    write_lookup_table_csv(rows, args.output)

    table = Table(title=f"benchmark lookup table -- {len(rows)} rows")
    for column in ("device_id", "backend", "level", "precision", "resolution", "end_to_end_p50_ms", "throughput_fps"):
        table.add_column(column)
    for row in rows:
        table.add_row(*(str(row.get(c, "")) for c in ("device_id", "backend", "level", "precision", "resolution", "end_to_end_p50_ms", "throughput_fps")))
    console.print(table)
    console.print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
