"""Turns the raw output of `scripts/benchmark/run_trtexec_protocol.sh` (3x
`run<N>_times.json` + `environment.txt`) into one `BenchmarkRecord`, pooling all three
independent runs' per-iteration samples before computing stats -- per
`docs/RESEARCH_PLAN.md` §9 rule 3 (three independent runs), pooling is the simplest
correct way to fold them into one CI without discarding any run's data.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from imavis_edge_seg.benchmark.parsers import parse_hailortcli_csv, parse_trtexec_export_times
from imavis_edge_seg.benchmark.report import BenchmarkRecord
from imavis_edge_seg.benchmark.stats import compute_latency_stats
from imavis_edge_seg.config import Backend, ElasticityLevel


def _parse_environment(env_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
        elif line.strip():
            env.setdefault("raw_environment_lines", "")
            env["raw_environment_lines"] += line.strip() + "; "
    return env


def build_record_from_trtexec_dir(
    output_dir: str | Path,
    device_id: str,
    backend: Backend,
    level: ElasticityLevel,
    precision: str,
    resolution: tuple[int, int],
) -> BenchmarkRecord:
    output_dir = Path(output_dir)
    run_files = sorted(output_dir.glob("run*_times.json"))
    if not run_files:
        raise FileNotFoundError(f"no run*_times.json found under {output_dir}")

    pooled_end_to_end: list[float] = []
    pooled_kernel_only: list[float] = []
    for run_file in run_files:
        end_to_end, kernel_only = parse_trtexec_export_times(run_file)
        pooled_end_to_end.extend(end_to_end)
        pooled_kernel_only.extend(kernel_only)

    env = _parse_environment(output_dir / "environment.txt")
    # trtexec's own banner looks like "[TensorRT v8502]" -- require the "v<digits>"
    # form so this doesn't also match unrelated "TensorRT.trtexec" substrings.
    match = re.search(r"TensorRT v(\d+)", env.get("raw_environment_lines", ""))
    toolchain = {
        "trtexec_version_line": env.get("raw_environment_lines", "")[:200],
        "trt_version": match.group(1) if match else "unknown",
    }

    end_to_end_stats = compute_latency_stats(pooled_end_to_end)
    kernel_only_stats = compute_latency_stats(pooled_kernel_only)

    return BenchmarkRecord(
        device_id=device_id,
        backend=backend,
        level=level,
        precision=precision,
        resolution=resolution,
        run_index=-1,  # pooled across all independent runs, not a single run
        timestamp_utc=env.get("timestamp_utc", datetime.now(UTC).isoformat()),
        toolchain=toolchain,
        end_to_end=end_to_end_stats,
        kernel_only=kernel_only_stats,
        throughput_fps=1000.0 / end_to_end_stats.mean_ms if end_to_end_stats.mean_ms > 0 else None,
        notes=(
            f"pooled from {len(run_files)} independent runs "
            f"({sum(1 for _ in pooled_end_to_end)} total iterations); "
            "throughput_fps derived as 1000/mean_latency_ms, not trtexec's own "
            "pipelined-throughput number -- see run*_log.txt for that"
        ),
    )


def build_record_from_hailo_dir(
    output_dir: str | Path,
    device_id: str,
    backend: Backend,
    level: ElasticityLevel,
    precision: str,
    resolution: tuple[int, int],
) -> BenchmarkRecord:
    """Builds a `BenchmarkRecord` from `scripts/benchmark/run_hailo_protocol.sh`'s output
    (3x `run<N>.csv` + `environment.txt`). Unlike trtexec, `hailortcli run --csv` reports
    one summary row per invocation, not one row per frame -- so here the 3 independent
    runs (§9 rule 3) themselves are the n=3 sample the stats/CI are computed over, not
    thousands of per-frame timings. A 3-sample bootstrap CI is wide; treat it as a rough
    bound, not a tight one.
    """
    output_dir = Path(output_dir)
    run_files = sorted(output_dir.glob("run*.csv"))
    if not run_files:
        raise FileNotFoundError(f"no run*.csv found under {output_dir}")

    end_to_end: list[float] = []
    kernel_only: list[float] = []
    temps: list[float] = []
    powers: list[float] = []
    for run_file in run_files:
        row = parse_hailortcli_csv(run_file)
        if row.get("overall_latency") is not None:
            end_to_end.append(float(row["overall_latency"]))  # type: ignore[arg-type]
        if row.get("hw_latency") is not None:
            kernel_only.append(float(row["hw_latency"]))  # type: ignore[arg-type]
        if row.get("average_temp") is not None:
            temps.append(float(row["average_temp"]))  # type: ignore[arg-type]
        if row.get("average_power") is not None:
            powers.append(float(row["average_power"]))  # type: ignore[arg-type]

    env = _parse_environment(output_dir / "environment.txt")
    raw_env = env.get("raw_environment_lines", "")
    # `_parse_environment` joins lines with "; " -- exclude ";" from the match so a
    # value at the end of a line doesn't swallow that separator (e.g. "4.23.0;").
    hailort_match = re.search(r"HailoRT-CLI version ([^\s;]+)", raw_env)
    firmware_match = re.search(r"Firmware Version: ([^\s;]+)", raw_env)
    board_match = re.search(r"Board Name:\s*([^\s;]+)", raw_env)
    toolchain = {
        "hailort_version": hailort_match.group(1) if hailort_match else "unknown",
        "firmware_version": firmware_match.group(1) if firmware_match else "unknown",
        "board_name": board_match.group(1) if board_match else "unknown",
    }

    end_to_end_stats = compute_latency_stats(end_to_end) if end_to_end else None
    kernel_only_stats = compute_latency_stats(kernel_only) if kernel_only else None

    # This device's Hailo-8 M.2 module has no on-board power/current sensor
    # (confirmed live 2026-09-10 -- `hailortcli run --measure-power` fails with
    # "Power measurement not supported"), so `powers` is always empty here; leave
    # power_mw unset rather than reporting a number this hardware cannot produce.
    power_mw = {"average": sum(powers) / len(powers)} if powers else None
    temp_c = {"average": sum(temps) / len(temps)} if temps else None

    return BenchmarkRecord(
        device_id=device_id,
        backend=backend,
        level=level,
        precision=precision,
        resolution=resolution,
        run_index=-1,  # pooled across all independent runs, not a single run
        timestamp_utc=env.get("timestamp_utc", datetime.now(UTC).isoformat()),
        toolchain=toolchain,
        end_to_end=end_to_end_stats,
        kernel_only=kernel_only_stats,
        throughput_fps=(
            1000.0 / end_to_end_stats.mean_ms
            if end_to_end_stats is not None and end_to_end_stats.mean_ms > 0
            else None
        ),
        power_mw=power_mw,
        temp_c=temp_c,
        notes=(
            f"pooled from {len(run_files)} independent hailortcli invocations "
            f"(n={len(end_to_end)} run-level samples, not per-frame -- see docstring); "
            "no power/current telemetry available on this Hailo-8 module"
            if power_mw is None
            else f"pooled from {len(run_files)} independent hailortcli invocations "
            f"(n={len(end_to_end)} run-level samples, not per-frame -- see docstring)"
        ),
    )
