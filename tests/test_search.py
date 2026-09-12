import pytest

from imavis_edge_seg.search.pareto import (
    ParetoPoint,
    build_pareto_points,
    is_dominated,
    pareto_frontier,
    pareto_frontiers_per_target,
    select_under_latency_budget,
)


def _point(level: str, device: str, backend: str, latency: float, miou: float) -> ParetoPoint:
    return ParetoPoint(
        level=level,  # type: ignore[arg-type]
        device_id=device,
        backend=backend,  # type: ignore[arg-type]
        precision="fp16",
        latency_ms=latency,
        miou=miou,
        dataset="cityscapes",
    )


def test_is_dominated_strictly_worse_on_both_axes() -> None:
    worse = _point("tiny", "E1", "hailo_hef", latency=10.0, miou=0.3)
    better = _point("small", "E1", "hailo_hef", latency=8.0, miou=0.4)
    assert is_dominated(worse, [worse, better])
    assert not is_dominated(better, [worse, better])


def test_is_dominated_not_dominated_when_tradeoff_exists() -> None:
    # cheaper but less accurate, and pricier but more accurate -- neither dominates
    cheap = _point("tiny", "E1", "hailo_hef", latency=4.0, miou=0.3)
    accurate = _point("large", "E1", "hailo_hef", latency=40.0, miou=0.47)
    assert not is_dominated(cheap, [cheap, accurate])
    assert not is_dominated(accurate, [cheap, accurate])


def test_pareto_frontier_drops_dominated_points_sorted_by_latency() -> None:
    dominated = _point("small", "E1", "hailo_hef", latency=10.0, miou=0.3)
    cheap = _point("tiny", "E1", "hailo_hef", latency=4.0, miou=0.3)
    accurate = _point("large", "E1", "hailo_hef", latency=40.0, miou=0.47)
    frontier = pareto_frontier([dominated, cheap, accurate])
    assert dominated not in frontier
    assert frontier == [cheap, accurate]  # sorted ascending by latency


def test_pareto_frontiers_per_target_groups_by_device_and_backend() -> None:
    e1_tiny = _point("tiny", "E1", "hailo_hef", latency=4.0, miou=0.3)
    e1_large = _point("large", "E1", "hailo_hef", latency=40.0, miou=0.47)
    e3_tiny = _point("tiny", "E3", "tensorrt_gpu", latency=0.9, miou=0.3)
    groups = pareto_frontiers_per_target([e1_tiny, e1_large, e3_tiny])
    assert set(groups.keys()) == {("E1", "hailo_hef"), ("E3", "tensorrt_gpu")}
    assert groups[("E1", "hailo_hef")] == [e1_tiny, e1_large]
    assert groups[("E3", "tensorrt_gpu")] == [e3_tiny]


def test_select_under_latency_budget_picks_best_miou_within_budget() -> None:
    points = [
        _point("tiny", "E1", "hailo_hef", latency=4.0, miou=0.30),
        _point("small", "E1", "hailo_hef", latency=7.0, miou=0.36),
        _point("medium", "E1", "hailo_hef", latency=20.0, miou=0.41),
        _point("large", "E1", "hailo_hef", latency=41.0, miou=0.47),
    ]
    assert select_under_latency_budget(points, budget_ms=10.0).level == "small"
    assert select_under_latency_budget(points, budget_ms=45.0).level == "large"
    assert select_under_latency_budget(points, budget_ms=1.0) is None


def test_select_under_latency_budget_ties_broken_by_lower_latency() -> None:
    a = _point("small", "E1", "hailo_hef", latency=7.0, miou=0.40)
    b = _point("medium", "E1", "hailo_hef", latency=20.0, miou=0.40)
    assert select_under_latency_budget([a, b], budget_ms=25.0) == a


def test_build_pareto_points_joins_lookup_rows_with_miou_by_level() -> None:
    # Real data, 2026-09-10/11 (reports/eval_pace_seg_v1_step100000.json,
    # outputs/benchmark_lookup_table.csv, large-level rows).
    lookup_rows = [
        {
            "device_id": "E1",
            "backend": "hailo_hef",
            "level": "large",
            "precision": "fp16",
            "end_to_end_mean_ms": 41.1363,
            "end_to_end_p95_ms": 41.18071,
        },
        {
            "device_id": "E3",
            "backend": "tensorrt_gpu",
            "level": "large",
            "precision": "fp16",
            "end_to_end_mean_ms": 9.38867,
            "end_to_end_p95_ms": 9.42,
        },
        {
            # no matching mIoU entry for "small" in miou_by_level below -- must be skipped
            "device_id": "E1",
            "backend": "hailo_hef",
            "level": "small",
            "precision": "fp16",
            "end_to_end_mean_ms": 6.6073,
            "end_to_end_p95_ms": 6.631906,
        },
    ]
    miou_by_level = {"large": {"cityscapes": 0.4712, "acdc/fog": 0.4919}}

    points = build_pareto_points(lookup_rows, miou_by_level, dataset="cityscapes")
    assert len(points) == 2  # "small" row skipped, no miou_by_level["small"]
    by_device = {p.device_id: p for p in points}
    assert by_device["E1"].miou == pytest.approx(0.4712)
    assert by_device["E1"].latency_ms == pytest.approx(41.18071)  # p95, the default field
    assert by_device["E3"].latency_ms == pytest.approx(9.42)


def test_build_pareto_points_honors_latency_field_override() -> None:
    lookup_rows = [
        {
            "device_id": "E1",
            "backend": "hailo_hef",
            "level": "large",
            "precision": "fp16",
            "end_to_end_mean_ms": 41.1363,
            "end_to_end_p95_ms": 41.18071,
        }
    ]
    miou_by_level = {"large": {"cityscapes": 0.4712}}
    points = build_pareto_points(
        lookup_rows, miou_by_level, dataset="cityscapes", latency_field="end_to_end_mean_ms"
    )
    assert points[0].latency_ms == pytest.approx(41.1363)


def test_build_pareto_points_skips_rows_with_missing_latency_value() -> None:
    lookup_rows = [
        {
            "device_id": "E1",
            "backend": "hailo_hef",
            "level": "large",
            "precision": "fp16",
            "end_to_end_mean_ms": None,
            "end_to_end_p95_ms": "",
        }
    ]
    miou_by_level = {"large": {"cityscapes": 0.4712}}
    assert build_pareto_points(lookup_rows, miou_by_level, dataset="cityscapes") == []
