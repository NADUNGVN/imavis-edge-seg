from pathlib import Path

import pytest

from imavis_edge_seg.benchmark.parsers import parse_hailortcli_csv, parse_trtexec_export_times
from imavis_edge_seg.benchmark.report import BenchmarkRecord, read_benchmark_record
from imavis_edge_seg.benchmark.stats import bootstrap_ci, compute_latency_stats

_FIXTURES = Path(__file__).parent / "fixtures"


def test_compute_latency_stats_basic_properties() -> None:
    samples = [1.0, 1.1, 0.9, 1.2, 0.95, 1.05, 1.0, 1.15, 0.85, 1.0]
    stats = compute_latency_stats(samples)
    assert stats.n == 10
    assert stats.min_ms <= stats.p50_ms <= stats.max_ms
    assert stats.p50_ms <= stats.p90_ms <= stats.p95_ms <= stats.p99_ms
    assert stats.ci95_low_ms <= stats.mean_ms <= stats.ci95_high_ms


def test_compute_latency_stats_rejects_empty() -> None:
    with pytest.raises(ValueError):
        compute_latency_stats([])


def test_bootstrap_ci_deterministic_given_seed() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci_a = bootstrap_ci(samples, seed=42)
    ci_b = bootstrap_ci(samples, seed=42)
    assert ci_a == ci_b


def test_bootstrap_ci_narrows_with_more_samples() -> None:
    import random

    rng = random.Random(0)
    small = [1.0 + rng.gauss(0, 0.1) for _ in range(10)]
    large = [1.0 + rng.gauss(0, 0.1) for _ in range(2000)]
    ci_small = bootstrap_ci(small, seed=1)
    ci_large = bootstrap_ci(large, seed=1)
    width_small = ci_small[1] - ci_small[0]
    width_large = ci_large[1] - ci_large[0]
    assert width_large < width_small


def test_parse_trtexec_export_times_real_fixture() -> None:
    end_to_end, kernel_only = parse_trtexec_export_times(
        _FIXTURES / "trtexec_export_times_sample.json"
    )
    assert len(end_to_end) == 25
    assert len(kernel_only) == 25
    assert all(v > 0 for v in end_to_end)
    assert all(v > 0 for v in kernel_only)
    # end-to-end (h2d+compute+d2h+enqueue) should never be less than kernel-only compute
    assert all(e >= k for e, k in zip(end_to_end, kernel_only, strict=True))


def test_parse_hailortcli_csv_real_header_format() -> None:
    row = parse_hailortcli_csv(_FIXTURES / "hailortcli_run_sample.csv")
    assert row["net_name"] == "pace_seg_tiny"
    assert row["status"] == 0.0
    assert row["fps"] == pytest.approx(593.70)
    assert row["num_of_frames"] == 2972
    assert row["average_power"] == pytest.approx(1912.4)
    assert row["min_current"] is None  # not measured in this sample -- must stay None, not 0


def test_benchmark_record_roundtrip(tmp_path: Path) -> None:
    stats = compute_latency_stats([1.0, 1.1, 0.9, 1.05])
    record = BenchmarkRecord(
        device_id="E3",
        backend="tensorrt_gpu",
        level="tiny",
        precision="fp16",
        resolution=(384, 192),
        run_index=0,
        timestamp_utc="2026-09-10T00:00:00Z",
        toolchain={"trt": "8.5.2.2", "jetpack": "5.1.4"},
        end_to_end=stats,
        kernel_only=stats,
        throughput_fps=850.0,
        notes="smoke test fixture",
    )
    path = tmp_path / "record.json"
    record.write(path)
    reloaded = read_benchmark_record(path)
    assert reloaded.device_id == "E3"
    assert reloaded.end_to_end is not None
    assert reloaded.end_to_end.mean_ms == pytest.approx(stats.mean_ms)
    assert reloaded.resolution == (384, 192)


def test_build_record_from_trtexec_dir(tmp_path: Path) -> None:
    import json

    from imavis_edge_seg.benchmark.aggregate import build_record_from_trtexec_dir

    fixture = json.loads((_FIXTURES / "trtexec_export_times_sample.json").read_text())
    for run in (1, 2, 3):
        (tmp_path / f"run{run}_times.json").write_text(json.dumps(fixture))
    # Real trtexec banner format, captured live on E3 2026-09-10 --
    # "TensorRT v8.5.2.2" never appears; it's always "[TensorRT v8502]".
    (tmp_path / "environment.txt").write_text(
        "device_id=E3\ntimestamp_utc=2026-09-10T00:00:00Z\n"
        "&&&& RUNNING TensorRT.trtexec [TensorRT v8502] # trtexec --help\n"
    )

    record = build_record_from_trtexec_dir(
        tmp_path,
        device_id="E3",
        backend="tensorrt_gpu",
        level="tiny",
        precision="fp16",
        resolution=(384, 192),
    )
    assert record.toolchain["trt_version"] == "8502"
    assert record.device_id == "E3"
    assert record.end_to_end is not None
    assert record.end_to_end.n == 25 * 3  # pooled across 3 runs
    assert record.kernel_only is not None
    assert record.throughput_fps is not None and record.throughput_fps > 0


def test_build_record_from_trtexec_dir_missing_runs_raises(tmp_path: Path) -> None:
    from imavis_edge_seg.benchmark.aggregate import build_record_from_trtexec_dir

    with pytest.raises(FileNotFoundError):
        build_record_from_trtexec_dir(
            tmp_path,
            device_id="E3",
            backend="tensorrt_gpu",
            level="tiny",
            precision="fp16",
            resolution=(384, 192),
        )


def test_build_lookup_table_merges_multiple_records(tmp_path: Path) -> None:
    from imavis_edge_seg.benchmark.lookup_table import build_lookup_table, write_lookup_table_csv

    stats = compute_latency_stats([1.0, 1.1, 0.9, 1.05])
    records_dir = tmp_path / "records"

    BenchmarkRecord(
        device_id="E3",
        backend="tensorrt_gpu",
        level="tiny",
        precision="fp16",
        resolution=(384, 192),
        run_index=-1,
        timestamp_utc="2026-09-10T00:00:00Z",
        toolchain={"trt_version": "8502"},
        end_to_end=stats,
        kernel_only=stats,
        throughput_fps=850.0,
    ).write(records_dir / "e3_tensorrt_tiny_fp16.json")

    BenchmarkRecord(
        device_id="E1",
        backend="hailo",
        level="tiny",
        precision="fp16",
        resolution=(384, 192),
        run_index=-1,
        timestamp_utc="2026-09-10T00:00:00Z",
        toolchain={"hailort": "4.17.0"},
        end_to_end=stats,
        throughput_fps=593.7,
        power_mw={"min": 1800.0, "average": 1912.4, "max": 2050.0},
        temp_c={"average": 52.3},
    ).write(records_dir / "e1_hailo_tiny_fp16.json")

    rows = build_lookup_table(records_dir)
    assert len(rows) == 2
    # sorted by (device_id, backend, level, precision, run_index) -- E1 before E3
    assert rows[0]["device_id"] == "E1"
    assert rows[0]["resolution"] == "384x192"
    # energy/frame only computable when both power and end-to-end latency are present
    assert rows[0]["energy_mj_per_frame"] == pytest.approx(1912.4 * stats.mean_ms / 1000.0)
    assert "energy_mj_per_frame" not in rows[1]  # E3 record has no power_mw

    csv_path = tmp_path / "lookup_table.csv"
    write_lookup_table_csv(rows, csv_path)
    lines = csv_path.read_text().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert "energy_mj_per_frame" in lines[0]


def test_build_lookup_table_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    from imavis_edge_seg.benchmark.lookup_table import build_lookup_table

    assert build_lookup_table(tmp_path) == []
