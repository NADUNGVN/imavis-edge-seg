import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from imavis_edge_seg.models.blocks import SlimmableConv2d  # noqa: E402
from imavis_edge_seg.search.flops import (  # noqa: E402
    FlopsPoint,
    build_flops_points,
    count_flops,
    evaluate_flops_proxy_at_budget,
    fit_flops_to_latency_rate,
    select_under_flops_budget,
)
from imavis_edge_seg.search.pareto import ParetoPoint  # noqa: E402


def test_count_flops_matches_hand_computed_value_for_plain_conv2d() -> None:
    # 3x3 conv, in=3, out=8, stride=1, padding=1 -> output spatial size == input (4x4).
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    x = torch.randn(1, 3, 4, 4)
    flops = count_flops(conv, lambda: conv(x))
    expected_macs = 3 * 8 * 3 * 3 * 4 * 4  # in_ch * out_ch * kh * kw * out_h * out_w
    assert flops == 2 * expected_macs


def test_count_flops_scales_with_output_spatial_size() -> None:
    conv = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)
    small = count_flops(conv, lambda: conv(torch.randn(1, 3, 8, 8)))
    large = count_flops(conv, lambda: conv(torch.randn(1, 3, 16, 16)))
    assert large == small * 4  # doubling H and W quadruples output pixel count


def test_count_flops_zero_for_a_model_with_no_conv_layers() -> None:
    model = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2))
    x = torch.randn(1, 3, 8, 8)
    assert count_flops(model, lambda: model(x)) == 0


def test_count_flops_slimmable_conv2d_uses_actual_active_channels_not_max() -> None:
    conv = SlimmableConv2d(max_in_channels=16, max_out_channels=16, kernel_size=3, padding=1)
    x_full = torch.randn(1, 16, 8, 8)
    x_half = torch.randn(1, 8, 8, 8)
    full_flops = count_flops(conv, lambda: conv(x_full, active_in=16, active_out=16))
    half_flops = count_flops(conv, lambda: conv(x_half, active_in=8, active_out=8))
    # Halving both active_in and active_out should quarter the FLOPs (quadratic in width).
    assert half_flops == pytest.approx(full_flops / 4)


def test_count_flops_slimmable_conv2d_depthwise_uses_groups_equal_to_active_out() -> None:
    conv = SlimmableConv2d(
        max_in_channels=8, max_out_channels=8, kernel_size=3, padding=1, depthwise=True
    )
    x = torch.randn(1, 4, 8, 8)
    flops = count_flops(conv, lambda: conv(x, active_in=4, active_out=4))
    # Depthwise: groups == out_channels, so in_channels/groups == 1 per output channel.
    expected_macs = 1 * 4 * 3 * 3 * 8 * 8
    assert flops == 2 * expected_macs


def test_count_flops_sums_across_multiple_conv_layers() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(4, 4, kernel_size=1, bias=False),
    )
    x = torch.randn(1, 3, 4, 4)
    flops = count_flops(model, lambda: model(x))
    per_layer = 3 * 4 * 1 * 1 * 4 * 4  # first layer's MACs
    second_layer = 4 * 4 * 1 * 1 * 4 * 4
    assert flops == 2 * (per_layer + second_layer)


def test_build_flops_points_joins_on_level_and_dataset() -> None:
    flops_by_level = {"tiny": 1000, "small": 4000}
    miou_by_level = {"tiny": {"cityscapes": 0.3}, "small": {"cityscapes": 0.36, "acdc/fog": 0.4}}
    points = build_flops_points(flops_by_level, miou_by_level, dataset="cityscapes")
    assert {p.level for p in points} == {"tiny", "small"}
    assert all(p.dataset == "cityscapes" for p in points)


def test_build_flops_points_skips_levels_missing_from_miou_table() -> None:
    flops_by_level = {"tiny": 1000, "small": 4000}
    miou_by_level = {"tiny": {"cityscapes": 0.3}}  # "small" missing
    points = build_flops_points(flops_by_level, miou_by_level, dataset="cityscapes")
    assert [p.level for p in points] == ["tiny"]


def _flops_point(level: str, flops: int, miou: float) -> FlopsPoint:
    return FlopsPoint(level=level, flops=flops, miou=miou, dataset="cityscapes")  # type: ignore[arg-type]


def test_select_under_flops_budget_picks_highest_miou_within_budget() -> None:
    points = [
        _flops_point("tiny", 1000, 0.30),
        _flops_point("small", 4000, 0.36),
        _flops_point("medium", 12000, 0.41),
        _flops_point("large", 30000, 0.47),
    ]
    best = select_under_flops_budget(points, budget_flops=5000)
    assert best is not None and best.level == "small"


def test_select_under_flops_budget_returns_none_when_nothing_fits() -> None:
    points = [_flops_point("large", 30000, 0.47)]
    assert select_under_flops_budget(points, budget_flops=100) is None


def test_fit_flops_to_latency_rate_recovers_exact_rate_on_noiseless_data() -> None:
    flops_by_level = {"tiny": 1000, "small": 4000, "large": 10000}
    true_rate = 0.002
    latency_by_level = {level: flops * true_rate for level, flops in flops_by_level.items()}
    fitted = fit_flops_to_latency_rate(flops_by_level, latency_by_level)
    assert fitted == pytest.approx(true_rate)


def test_fit_flops_to_latency_rate_only_uses_overlapping_levels() -> None:
    flops_by_level = {"tiny": 1000, "small": 4000, "medium": 8000}
    latency_by_level = {"tiny": 2.0, "small": 8.0}  # "medium" missing -- must be ignored
    fitted = fit_flops_to_latency_rate(flops_by_level, latency_by_level)
    assert fitted == pytest.approx(0.002)


def test_fit_flops_to_latency_rate_returns_zero_for_no_overlap() -> None:
    assert fit_flops_to_latency_rate({"tiny": 1000}, {"large": 5.0}) == 0.0


# ---- evaluate_flops_proxy_at_budget (RQ1 budget sweep) --------------------------------


def _real_points() -> list[ParetoPoint]:
    return [
        ParetoPoint(level="tiny", device_id="E_TEST", backend="tensorrt_gpu", precision="fp16", latency_ms=1.0, miou=0.30, dataset="cityscapes"),  # type: ignore[arg-type]
        ParetoPoint(level="small", device_id="E_TEST", backend="tensorrt_gpu", precision="fp16", latency_ms=2.0, miou=0.40, dataset="cityscapes"),  # type: ignore[arg-type]
        ParetoPoint(level="medium", device_id="E_TEST", backend="tensorrt_gpu", precision="fp16", latency_ms=5.0, miou=0.50, dataset="cityscapes"),  # type: ignore[arg-type]
        ParetoPoint(level="large", device_id="E_TEST", backend="tensorrt_gpu", precision="fp16", latency_ms=10.0, miou=0.60, dataset="cityscapes"),  # type: ignore[arg-type]
    ]


def test_evaluate_flops_proxy_at_budget_matches_when_proxy_predicts_correctly() -> None:
    real_points = _real_points()
    perfect_proxy = {p.level: p.latency_ms for p in real_points}
    result = evaluate_flops_proxy_at_budget(real_points, perfect_proxy, budget_ms=3.5)
    assert result.real_level == result.proxy_level == "small"
    assert not result.mis_selected
    assert result.accuracy_regret == pytest.approx(0.0)
    assert result.slack_ms == pytest.approx(3.5 - 2.0)


def test_evaluate_flops_proxy_at_budget_detects_misselection_and_budget_violation() -> None:
    real_points = _real_points()
    # Proxy badly overestimates "small" and underestimates "medium" -- at budget=3.5
    # the real device should pick "small" (real 2.0ms), but the proxy picks "medium"
    # (predicted 3.0ms fits the budget; medium's REAL latency, 5.0ms, does not).
    bad_proxy = {"tiny": 1.0, "small": 5.0, "medium": 3.0, "large": 20.0}
    result = evaluate_flops_proxy_at_budget(real_points, bad_proxy, budget_ms=3.5)
    assert result.real_level == "small"
    assert result.proxy_level == "medium"
    assert result.mis_selected
    assert result.proxy_real_latency_ms == pytest.approx(5.0)
    assert result.slack_ms == pytest.approx(3.5 - 5.0)
    assert result.slack_ms is not None and result.slack_ms < 0  # proxy's pick violates the real budget
    assert result.accuracy_regret == pytest.approx(0.40 - 0.50)  # real(small) - proxy's real(medium)


def test_evaluate_flops_proxy_at_budget_proxy_refuses_service_real_could_serve() -> None:
    real_points = _real_points()
    impossible_proxy = {level: 1000.0 for level in ("tiny", "small", "medium", "large")}
    result = evaluate_flops_proxy_at_budget(real_points, impossible_proxy, budget_ms=3.5)
    assert result.real_level == "small"
    assert result.proxy_level is None
    assert result.mis_selected
    assert result.accuracy_regret is None
    assert result.slack_ms is None


def test_evaluate_flops_proxy_at_budget_both_infeasible_is_not_a_misselection() -> None:
    real_points = _real_points()
    proxy = {p.level: p.latency_ms for p in real_points}
    result = evaluate_flops_proxy_at_budget(real_points, proxy, budget_ms=0.1)  # below even "tiny"
    assert result.real_level is None
    assert result.proxy_level is None
    assert not result.mis_selected  # None == None -- both correctly refuse, not a disagreement
