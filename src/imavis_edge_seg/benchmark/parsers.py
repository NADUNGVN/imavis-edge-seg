"""Parsers for vendor benchmark tool output -- `trtexec --exportTimes` JSON (TensorRT
GPU/DLA) and `hailortcli run --csv` (Hailo). Deliberately reuse each vendor's own
measurement tool rather than reimplementing device-level timing, per
`docs/RESEARCH_PLAN.md` §9 rule 5 (disclose exact toolchain) -- these parsers only turn
that tool's own output into this project's common `BenchmarkRecord` shape.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def parse_trtexec_export_times(json_path: str | Path) -> tuple[list[float], list[float]]:
    """Returns (end_to_end_ms, kernel_only_ms) -- `latencyMs` (h2d+compute+d2h+enqueue
    overhead, the end-to-end number) and `computeMs` (kernel-only, GPU/DLA compute time)
    per iteration, from a `trtexec --exportTimes=<json_path>` run. Per §9 rule 7, these
    must be reported separately, never mixed."""
    records = json.loads(Path(json_path).read_text())
    end_to_end = [float(r["latencyMs"]) for r in records]
    kernel_only = [float(r["computeMs"]) for r in records]
    return end_to_end, kernel_only


def parse_hailortcli_csv(csv_path: str | Path) -> dict[str, float | str | None]:
    """Returns the single summary row from `hailortcli run --csv=<csv_path>
    --measure-latency --measure-power --measure-temp` as a dict of its columns.
    HailoRT reports one summary line per run (not per-frame timings like trtexec), so
    there is no distribution to bootstrap here -- multiple independent `hailortcli run`
    invocations (per §9 rule 3) are the unit of repetition instead."""
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")
    row = rows[0]
    result: dict[str, float | str | None] = {}
    for key, value in row.items():
        if value in (None, ""):
            result[key] = None
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result
