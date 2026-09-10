"""Merges many `BenchmarkRecord` JSON files (one per device/backend/level/precision
measurement, written by `report.py`) into one flat latency/energy lookup table --
`docs/RESEARCH_PLAN.md` §5.3 B / §5.4's `L_hat`/`E_hat` surrogates are meant to be fit
from a table like this, not from any single record in isolation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from imavis_edge_seg.benchmark.report import read_benchmark_record

# Column order for the CSV -- kept stable so a re-generated table diffs cleanly.
_COLUMNS = [
    "device_id",
    "backend",
    "level",
    "precision",
    "resolution",
    "run_index",
    "timestamp_utc",
    "git_commit",
    "end_to_end_n",
    "end_to_end_mean_ms",
    "end_to_end_p50_ms",
    "end_to_end_p95_ms",
    "end_to_end_p99_ms",
    "end_to_end_ci95_low_ms",
    "end_to_end_ci95_high_ms",
    "kernel_only_mean_ms",
    "kernel_only_p50_ms",
    "kernel_only_p95_ms",
    "kernel_only_p99_ms",
    "throughput_fps",
    "power_avg_mw",
    "energy_mj_per_frame",
    "temp_avg_c",
    "notes",
]


def _row_from_record(record_path: Path) -> dict[str, Any]:
    record = read_benchmark_record(record_path)
    row: dict[str, Any] = {
        "device_id": record.device_id,
        "backend": record.backend,
        "level": record.level,
        "precision": record.precision,
        "resolution": f"{record.resolution[0]}x{record.resolution[1]}",
        "run_index": record.run_index,
        "timestamp_utc": record.timestamp_utc,
        "git_commit": record.git_commit,
        "notes": record.notes,
    }

    if record.end_to_end is not None:
        row["end_to_end_n"] = record.end_to_end.n
        row["end_to_end_mean_ms"] = record.end_to_end.mean_ms
        row["end_to_end_p50_ms"] = record.end_to_end.p50_ms
        row["end_to_end_p95_ms"] = record.end_to_end.p95_ms
        row["end_to_end_p99_ms"] = record.end_to_end.p99_ms
        row["end_to_end_ci95_low_ms"] = record.end_to_end.ci95_low_ms
        row["end_to_end_ci95_high_ms"] = record.end_to_end.ci95_high_ms

    if record.kernel_only is not None:
        row["kernel_only_mean_ms"] = record.kernel_only.mean_ms
        row["kernel_only_p50_ms"] = record.kernel_only.p50_ms
        row["kernel_only_p95_ms"] = record.kernel_only.p95_ms
        row["kernel_only_p99_ms"] = record.kernel_only.p99_ms

    row["throughput_fps"] = record.throughput_fps

    power_avg = record.power_mw.get("average") if record.power_mw else None
    row["power_avg_mw"] = power_avg
    # J/frame needs an end-to-end latency; skip (leave unset) if either input is missing
    # rather than silently reporting 0 -- an absent number must stay absent.
    if power_avg is not None and record.end_to_end is not None:
        row["energy_mj_per_frame"] = power_avg * record.end_to_end.mean_ms / 1000.0

    if record.temp_c:
        row["temp_avg_c"] = record.temp_c.get("average")

    return row


def build_lookup_table(records_dir: str | Path) -> list[dict[str, Any]]:
    """Reads every `*.json` `BenchmarkRecord` under `records_dir` (recursively) and
    returns one flat row per record, sorted by (device_id, backend, level, precision,
    run_index) so the table reads the same regardless of filesystem iteration order."""
    records_dir = Path(records_dir)
    rows = [_row_from_record(path) for path in sorted(records_dir.rglob("*.json"))]
    rows.sort(
        key=lambda r: (
            r["device_id"],
            r["backend"],
            r["level"],
            r["precision"],
            r["run_index"],
        )
    )
    return rows


def write_lookup_table_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
