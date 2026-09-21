import numpy as np
import pytest

from imavis_edge_seg.router.grid import macro_quantile_grid, select_budget_matched_operating_point

# ---- macro_quantile_grid -----------------------------------------------------------------------


def test_macro_quantile_grid_single_split_matches_plain_quantile() -> None:
    values = np.linspace(0, 1, 101)  # 0.00, 0.01, ..., 1.00
    grid = macro_quantile_grid([values], quantiles=(0.5,))
    assert grid == pytest.approx([0.5], abs=1e-6)


def test_macro_quantile_grid_averages_across_splits_with_equal_weight() -> None:
    # Split A: 900 copies of 0.0 (would dominate a pooled/micro quantile).
    # Split B: 100 copies of 1.0.
    # Macro (average of each split's own median): (0.0 + 1.0) / 2 = 0.5, NOT the
    # pooled median (which would be ~0.0, since split A has 9x more samples).
    split_a = np.zeros(900)
    split_b = np.ones(100)
    grid = macro_quantile_grid([split_a, split_b], quantiles=(0.5,))
    assert grid == pytest.approx([0.5], abs=1e-6)


def test_macro_quantile_grid_returns_sorted_deduplicated_values() -> None:
    values = np.linspace(0, 1, 101)
    grid = macro_quantile_grid([values], quantiles=(0.1, 0.5, 0.5, 0.9))  # deliberate duplicate
    assert grid == sorted(grid)
    assert len(grid) == 3  # 0.5 collapses to one point


def test_macro_quantile_grid_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        macro_quantile_grid([])


# ---- select_budget_matched_operating_point -----------------------------------------------------------------------


def test_select_budget_matched_picks_highest_quality_within_budget() -> None:
    points = [
        (5.0, 0.60, "cheap_low_quality"),
        (8.0, 0.80, "mid_high_quality"),
        (20.0, 0.95, "expensive_best_quality"),
    ]
    label, feasible = select_budget_matched_operating_point(points, budget_ms=10.0)
    assert label == "mid_high_quality"
    assert feasible is True


def test_select_budget_matched_ties_broken_by_lower_latency() -> None:
    points = [
        (5.0, 0.80, "cheaper"),
        (9.0, 0.80, "pricier"),  # same quality, more latency
    ]
    label, feasible = select_budget_matched_operating_point(points, budget_ms=10.0)
    assert label == "cheaper"
    assert feasible is True


def test_select_budget_matched_falls_back_to_cheapest_when_nothing_fits() -> None:
    points = [
        (50.0, 0.60, "a"),
        (100.0, 0.90, "b"),
    ]
    label, feasible = select_budget_matched_operating_point(points, budget_ms=10.0)
    assert label == "a"  # cheapest overall, even though it's over budget
    assert feasible is False


def test_select_budget_matched_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        select_budget_matched_operating_point([], budget_ms=10.0)
