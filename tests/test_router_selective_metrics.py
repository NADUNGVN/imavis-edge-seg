import numpy as np
import pytest

from imavis_edge_seg.router.selective_metrics import (
    area_under_risk_coverage,
    budget_violation_rate,
    risk_at_coverage,
)

# ---- risk_at_coverage -----------------------------------------------------------------------


def test_risk_at_coverage_full_coverage_is_plain_mean() -> None:
    risks = [0.3, 0.1, 0.5, 0.2]
    errors = [0.6, 0.2, 1.0, 0.4]
    assert risk_at_coverage(risks, errors, coverage=1.0) == pytest.approx(np.mean(errors))


def test_risk_at_coverage_uses_lowest_risk_samples_only() -> None:
    # Sorted by risk ascending: risk=0.1->error=0.2, risk=0.2->error=0.4, ...
    risks = [0.3, 0.1, 0.5, 0.2]
    errors = [0.6, 0.2, 1.0, 0.4]
    # coverage=0.5 -> the 2 lowest-risk samples: errors 0.2 and 0.4 -> mean 0.3.
    assert risk_at_coverage(risks, errors, coverage=0.5) == pytest.approx(0.3)


def test_risk_at_coverage_perfect_signal_has_low_error_at_low_coverage() -> None:
    rng = np.random.default_rng(0)
    n = 200
    errors = rng.uniform(0, 1, size=n)
    risks = errors.copy()  # risk score == error exactly -> perfect ranking
    low_coverage = risk_at_coverage(risks, errors, coverage=0.1)
    full_coverage = risk_at_coverage(risks, errors, coverage=1.0)
    assert low_coverage < full_coverage


def test_risk_at_coverage_rejects_out_of_range_coverage() -> None:
    with pytest.raises(ValueError):
        risk_at_coverage([0.1], [0.1], coverage=0.0)
    with pytest.raises(ValueError):
        risk_at_coverage([0.1], [0.1], coverage=1.1)


def test_risk_at_coverage_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        risk_at_coverage([], [], coverage=0.5)


def test_risk_at_coverage_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        risk_at_coverage([0.1, 0.2], [0.1], coverage=0.5)


# ---- area_under_risk_coverage -----------------------------------------------------------------------


def test_aurc_perfect_signal_scores_lower_than_random_signal() -> None:
    rng = np.random.default_rng(0)
    n = 300
    errors = rng.uniform(0, 1, size=n)
    perfect_risk = errors.copy()
    random_risk = rng.permutation(errors)  # same error distribution, uncorrelated ranking

    perfect_aurc = area_under_risk_coverage(perfect_risk, errors)
    random_aurc = area_under_risk_coverage(random_risk, errors)
    assert perfect_aurc < random_aurc


def test_aurc_constant_error_equals_that_constant_regardless_of_risk_ranking() -> None:
    risks = [0.9, 0.1, 0.5, 0.3]
    errors = [0.42, 0.42, 0.42, 0.42]
    assert area_under_risk_coverage(risks, errors) == pytest.approx(0.42)


def test_aurc_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        area_under_risk_coverage([], [])


# ---- budget_violation_rate -----------------------------------------------------------------------


def test_budget_violation_rate_none_over_budget_is_zero() -> None:
    assert budget_violation_rate([1.0, 2.0, 3.0], budget_ms=5.0) == pytest.approx(0.0)


def test_budget_violation_rate_counts_strictly_over_budget() -> None:
    # 5.0 exactly at budget -> not a violation; 6.0 and 7.0 -> violations (2 of 4).
    assert budget_violation_rate([1.0, 5.0, 6.0, 7.0], budget_ms=5.0) == pytest.approx(0.5)


def test_budget_violation_rate_all_over_budget_is_one() -> None:
    assert budget_violation_rate([10.0, 20.0], budget_ms=5.0) == pytest.approx(1.0)


def test_budget_violation_rate_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        budget_violation_rate([], budget_ms=5.0)
