import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imavis_edge_seg.config import RouterConfig  # noqa: E402
from imavis_edge_seg.data.labels import IGNORE_INDEX  # noqa: E402
from imavis_edge_seg.router.calibrator import (  # noqa: E402
    RiskCalibrator,
    fit_per_level_calibrators,
    fit_risk_calibrator,
    prediction_inversion_rate,
)
from imavis_edge_seg.router.observed_error import compute_per_image_error  # noqa: E402
from imavis_edge_seg.router.policy import select_level  # noqa: E402
from imavis_edge_seg.router.risk_probe import (  # noqa: E402
    compute_deployment_risk_score,
    compute_risk_score,
)
from imavis_edge_seg.search.pareto import ParetoPoint  # noqa: E402

# ---- risk_probe ---------------------------------------------------------------------


def test_compute_risk_score_confident_prediction_scores_lower_than_uniform() -> None:
    confident_logits = torch.zeros(1, 4, 2, 2)
    confident_logits[:, 0] = 20.0  # near-one-hot after softmax -> low entropy
    uniform_logits = torch.zeros(1, 4, 2, 2)  # all-equal logits -> max-entropy softmax

    confident_score = compute_risk_score(confident_logits)
    uniform_score = compute_risk_score(uniform_logits)

    assert confident_score.shape == (1,)
    assert float(confident_score) < float(uniform_score)


def test_compute_risk_score_ignore_index_pixels_excluded() -> None:
    logits = torch.zeros(1, 4, 2, 2)
    logits[:, 0] = 20.0  # confident everywhere

    clean_target = torch.zeros(1, 2, 2, dtype=torch.long)
    clean_score = compute_risk_score(logits, clean_target)

    # Same logits, but half the target is IGNORE_INDEX with (irrelevantly) different
    # values there -- since those pixels are excluded, the score must be identical.
    target_with_ignore = clean_target.clone()
    target_with_ignore[:, :, 1] = IGNORE_INDEX
    ignore_score = compute_risk_score(logits, target_with_ignore)

    assert float(clean_score) == pytest.approx(float(ignore_score), abs=1e-6)


def test_compute_risk_score_batch_dimension_independent() -> None:
    logits = torch.randn(3, 5, 4, 4)
    scores = compute_risk_score(logits)
    assert scores.shape == (3,)


def test_deployment_risk_score_is_identical_for_fit_and_inference() -> None:
    logits = torch.zeros(1, 3, 2, 2)
    logits[:, 0] = 12.0
    # Make one pixel maximally uncertain. A ground-truth mask could exclude it,
    # but the deployed feature must include it at both fit and inference time.
    logits[:, :, 1, 1] = 0.0
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    target[:, 1, 1] = IGNORE_INDEX

    deployment_feature_at_fit = compute_deployment_risk_score(logits)
    deployment_feature_at_inference = compute_deployment_risk_score(logits)
    legacy_masked_feature = compute_risk_score(logits, target)

    assert torch.equal(deployment_feature_at_fit, deployment_feature_at_inference)
    assert torch.equal(deployment_feature_at_fit, compute_risk_score(logits))
    assert not torch.allclose(deployment_feature_at_fit, legacy_masked_feature)


# ---- calibrator -----------------------------------------------------------------------


def test_fit_risk_calibrator_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        fit_risk_calibrator([], [])


def test_fit_risk_calibrator_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_risk_calibrator([1.0, 2.0], [0.1])


def test_fit_risk_calibrator_higher_raw_score_gives_higher_or_equal_expected_error() -> None:
    rng = np.random.default_rng(0)
    raw_scores = rng.uniform(0, 3, size=500)
    # True relationship: error increases with raw_score, plus noise.
    errors = np.clip(raw_scores / 3 + rng.normal(0, 0.05, size=500), 0, 1)

    calibrator = fit_risk_calibrator(raw_scores, errors, num_bins=10)
    low = calibrator.predict(0.1)
    mid = calibrator.predict(1.5)
    high = calibrator.predict(2.9)
    assert low <= mid <= high


def test_fit_risk_calibrator_enforces_monotonicity_despite_noisy_bins() -> None:
    # Deliberately non-monotonic raw bin averages (a dip in the middle) -- the fitted
    # calibrator's predictions must still never decrease as raw_score increases.
    raw_scores = np.array([0.0] * 10 + [1.0] * 10 + [2.0] * 10)
    errors = np.array([0.1] * 10 + [0.05] * 10 + [0.3] * 10)  # dips at raw_score=1.0
    calibrator = fit_risk_calibrator(raw_scores, errors, num_bins=3)
    assert np.all(np.diff(calibrator.bin_expected_error) >= 0)


def test_fit_risk_calibrator_predict_out_of_range_clamps_to_edge_bins() -> None:
    raw_scores = np.linspace(0, 1, 100)
    errors = raw_scores.copy()
    calibrator = fit_risk_calibrator(raw_scores, errors, num_bins=5)
    assert calibrator.predict(-10.0) == calibrator.predict(0.0)
    assert calibrator.predict(100.0) == calibrator.predict(1.0)


# ---- policy -----------------------------------------------------------------------


def _point(level: str, latency: float, miou: float) -> ParetoPoint:
    return ParetoPoint(
        level=level,  # type: ignore[arg-type]
        device_id="E1",
        backend="hailo_hef",  # type: ignore[arg-type]
        precision="fp16",
        latency_ms=latency,
        miou=miou,
        dataset="cityscapes",
    )


_CANDIDATES = [
    _point("tiny", 4.0, 0.30),
    _point("small", 7.0, 0.36),
    _point("medium", 20.0, 0.41),
    _point("large", 41.0, 0.47),
]


def test_select_level_raises_on_empty_candidates() -> None:
    with pytest.raises(ValueError):
        select_level([], RouterConfig(strategy="static_small"))


def test_select_level_static_small_and_large() -> None:
    assert select_level(_CANDIDATES, RouterConfig(strategy="static_small")) == "tiny"
    assert select_level(_CANDIDATES, RouterConfig(strategy="static_large")) == "large"


def test_select_level_oracle_picks_best_miou_regardless_of_cost() -> None:
    assert select_level(_CANDIDATES, RouterConfig(strategy="oracle")) == "large"


def test_select_level_entropy_requires_raw_risk() -> None:
    with pytest.raises(ValueError):
        select_level(_CANDIDATES, RouterConfig(strategy="entropy"))


def test_select_level_calibrated_risk_requires_calibrated_risk() -> None:
    with pytest.raises(ValueError):
        select_level(_CANDIDATES, RouterConfig(strategy="calibrated_risk"))


def test_select_level_low_risk_picks_smallest() -> None:
    config = RouterConfig(strategy="calibrated_risk", risk_target=0.05)
    level = select_level(_CANDIDATES, config, calibrated_risk=0.02)
    assert level == "tiny"


def test_select_level_high_risk_escalates_to_largest() -> None:
    config = RouterConfig(strategy="calibrated_risk", risk_target=0.05)
    level = select_level(_CANDIDATES, config, calibrated_risk=1.0)  # 20x over target
    assert level == "large"


def test_select_level_moderate_risk_picks_a_middle_level() -> None:
    config = RouterConfig(strategy="calibrated_risk", risk_target=0.05)
    # 2x over target -> escalate 2 steps from the smallest (index 2 -> "medium").
    level = select_level(_CANDIDATES, config, calibrated_risk=0.11)
    assert level == "medium"


def test_select_level_unknown_strategy_raises() -> None:
    config = RouterConfig.model_construct(strategy="not_a_real_strategy", risk_target=0.05, enabled=True, window_frames=16)
    with pytest.raises(ValueError):
        select_level(_CANDIDATES, config, calibrated_risk=0.1)


# ---- policy: cell B (latency_spacing_risk) -----------------------------------------------------------------------


def test_select_level_latency_spacing_low_risk_picks_cheapest() -> None:
    config = RouterConfig(strategy="latency_spacing_risk", risk_target=0.05)
    assert select_level(_CANDIDATES, config, calibrated_risk=0.01) == "tiny"


def test_select_level_latency_spacing_targets_latency_magnitude_not_rank() -> None:
    # risk_target=0.05, risk=0.25 -> r=5 -> t_target=clip(4*5, 4, 41)=20, an exact
    # match to "medium"'s latency (20.0) -- this is the case _select_by_risk (rank
    # stepping) would also reach index min(5, 3)="large" instead, so this is a real
    # behavioral difference from cell A, not just a relabeling.
    config = RouterConfig(strategy="latency_spacing_risk", risk_target=0.05)
    assert select_level(_CANDIDATES, config, calibrated_risk=0.25) == "medium"


def test_select_level_latency_spacing_very_high_risk_caps_at_largest() -> None:
    config = RouterConfig(strategy="latency_spacing_risk", risk_target=0.05)
    assert select_level(_CANDIDATES, config, calibrated_risk=10.0) == "large"


def test_select_level_latency_spacing_ties_broken_toward_cheaper_candidate() -> None:
    candidates = [
        _point("a", 10.0, 0.3),
        _point("b", 14.0, 0.3),  # both equidistant (|10-12|==|14-12|==2) from t_target=12
    ]
    config = RouterConfig(strategy="latency_spacing_risk", risk_target=1.0)
    assert select_level(candidates, config, calibrated_risk=1.2) == "a"  # type: ignore[arg-type]


def test_select_level_latency_spacing_requires_calibrated_risk() -> None:
    with pytest.raises(ValueError):
        select_level(_CANDIDATES, RouterConfig(strategy="latency_spacing_risk"))


def test_select_level_latency_spacing_rejects_non_positive_risk_target() -> None:
    config = RouterConfig(strategy="latency_spacing_risk", risk_target=0.0)
    with pytest.raises(ValueError):
        select_level(_CANDIDATES, config, calibrated_risk=0.1)


# ---- policy: cell C (candidate_specific_risk) -----------------------------------------------------------------------


def test_select_level_candidate_specific_picks_cheapest_qualifying_candidate() -> None:
    config = RouterConfig(strategy="candidate_specific_risk", risk_target=0.05)
    per_level_risk = {"tiny": 0.20, "small": 0.03, "medium": 0.01, "large": 0.005}
    assert select_level(_CANDIDATES, config, per_level_risk=per_level_risk) == "small"  # type: ignore[arg-type]


def test_select_level_candidate_specific_uses_each_candidates_own_risk() -> None:
    # small's own predicted risk is high even though a *scalar* single-probe risk
    # would have escalated past it -- medium is the first (cheapest) to qualify.
    config = RouterConfig(strategy="candidate_specific_risk", risk_target=0.05)
    per_level_risk = {"tiny": 0.20, "small": 0.20, "medium": 0.01, "large": 0.005}
    assert select_level(_CANDIDATES, config, per_level_risk=per_level_risk) == "medium"  # type: ignore[arg-type]


def test_select_level_candidate_specific_falls_back_to_largest_if_none_qualify() -> None:
    config = RouterConfig(strategy="candidate_specific_risk", risk_target=0.05)
    per_level_risk = {"tiny": 0.5, "small": 0.4, "medium": 0.3, "large": 0.2}
    assert select_level(_CANDIDATES, config, per_level_risk=per_level_risk) == "large"  # type: ignore[arg-type]


def test_select_level_candidate_specific_requires_per_level_risk() -> None:
    with pytest.raises(ValueError):
        select_level(_CANDIDATES, RouterConfig(strategy="candidate_specific_risk"))


# ---- policy: cell D (risk_latency_constrained) -----------------------------------------------------------------------


def test_select_level_risk_latency_constrained_picks_cheapest_qualifying_in_budget() -> None:
    config = RouterConfig(strategy="risk_latency_constrained", risk_target=0.05, latency_budget_ms=41.0)
    per_level_risk = {"tiny": 0.20, "small": 0.03, "medium": 0.01, "large": 0.005}
    assert select_level(_CANDIDATES, config, per_level_risk=per_level_risk) == "small"  # type: ignore[arg-type]


def test_select_level_risk_latency_constrained_picks_lowest_risk_in_budget_when_none_qualify() -> None:
    config = RouterConfig(strategy="risk_latency_constrained", risk_target=0.05, latency_budget_ms=41.0)
    per_level_risk = {"tiny": 0.5, "small": 0.4, "medium": 0.3, "large": 0.2}
    assert select_level(_CANDIDATES, config, per_level_risk=per_level_risk) == "large"  # type: ignore[arg-type]


def test_select_level_risk_latency_constrained_ties_broken_by_lower_latency() -> None:
    config = RouterConfig(strategy="risk_latency_constrained", risk_target=0.05, latency_budget_ms=41.0)
    per_level_risk = {"tiny": 0.5, "small": 0.4, "medium": 0.2, "large": 0.2}
    assert select_level(_CANDIDATES, config, per_level_risk=per_level_risk) == "medium"  # type: ignore[arg-type]


def test_select_level_risk_latency_constrained_excludes_candidates_over_budget() -> None:
    # large (41.0, risk 0.005) is excluded by the budget (20.0) even though it's the
    # lowest-risk candidate overall -- small (7.0, risk 0.03) and medium (20.0, risk
    # 0.01) both still qualify (<= 0.05) within budget, and small is the cheaper of
    # the two, so it wins (rule 1: cheapest among those meeting the target).
    config = RouterConfig(strategy="risk_latency_constrained", risk_target=0.05, latency_budget_ms=20.0)
    per_level_risk = {"tiny": 0.20, "small": 0.03, "medium": 0.01, "large": 0.005}
    assert select_level(_CANDIDATES, config, per_level_risk=per_level_risk) == "small"  # type: ignore[arg-type]


def test_select_level_risk_latency_constrained_falls_back_to_cheapest_when_budget_excludes_everything() -> None:
    config = RouterConfig(strategy="risk_latency_constrained", risk_target=0.05, latency_budget_ms=2.0)
    per_level_risk = {"tiny": 0.20, "small": 0.03, "medium": 0.01, "large": 0.005}
    assert select_level(_CANDIDATES, config, per_level_risk=per_level_risk) == "tiny"  # type: ignore[arg-type]


def test_select_level_risk_latency_constrained_requires_per_level_risk() -> None:
    config = RouterConfig(strategy="risk_latency_constrained", risk_target=0.05, latency_budget_ms=41.0)
    with pytest.raises(ValueError):
        select_level(_CANDIDATES, config)


def test_select_level_risk_latency_constrained_requires_latency_budget() -> None:
    config = RouterConfig(strategy="risk_latency_constrained", risk_target=0.05)
    with pytest.raises(ValueError):
        select_level(_CANDIDATES, config, per_level_risk={"tiny": 0.01})  # type: ignore[arg-type]


# ---- calibrator: per-level (cell C) -----------------------------------------------------------------------


def test_fit_per_level_calibrators_fits_one_independent_calibrator_per_level() -> None:
    rng = np.random.default_rng(0)
    raw_scores = rng.uniform(0, 3, size=200)
    per_level_errors = {
        "tiny": np.clip(raw_scores / 3 + rng.normal(0, 0.02, size=200), 0, 1),
        "large": np.clip(raw_scores / 10 + rng.normal(0, 0.02, size=200), 0, 1),  # much lower error overall
    }
    calibrators = fit_per_level_calibrators(raw_scores, per_level_errors, num_bins=5)
    assert set(calibrators) == {"tiny", "large"}
    # Same raw_score, different targets -> different fitted predictions.
    assert calibrators["tiny"].predict(2.9) > calibrators["large"].predict(2.9)


def test_fit_per_level_calibrators_each_calibrator_independently_monotonic() -> None:
    raw_scores = np.array([0.0] * 10 + [1.0] * 10 + [2.0] * 10)
    per_level_errors = {
        "tiny": np.array([0.1] * 10 + [0.05] * 10 + [0.3] * 10),  # dip at raw_score=1.0
        "small": np.array([0.0] * 10 + [0.0] * 10 + [0.1] * 10),
    }
    calibrators = fit_per_level_calibrators(raw_scores, per_level_errors, num_bins=3)
    for calibrator in calibrators.values():
        assert np.all(np.diff(calibrator.bin_expected_error) >= 0)


def test_prediction_inversion_rate_all_consistent_is_zero() -> None:
    ordered = ["tiny", "small", "large"]
    predicted = [
        {"tiny": 0.3, "small": 0.2, "large": 0.1},
        {"tiny": 0.5, "small": 0.3, "large": 0.05},
    ]
    assert prediction_inversion_rate(ordered, predicted) == pytest.approx(0.0)


def test_prediction_inversion_rate_counts_adjacent_pairs_where_bigger_is_worse() -> None:
    ordered = ["tiny", "small", "large"]
    # tiny->small: 0.2 > 0.3? no. small->large: 0.4 > 0.2? yes -- 1 of 2 pairs inverted.
    predicted = [{"tiny": 0.3, "small": 0.2, "large": 0.4}]
    assert prediction_inversion_rate(ordered, predicted) == pytest.approx(0.5)


def test_prediction_inversion_rate_rejects_fewer_than_two_levels() -> None:
    with pytest.raises(ValueError):
        prediction_inversion_rate(["tiny"], [{"tiny": 0.1}])


def test_prediction_inversion_rate_rejects_zero_images() -> None:
    with pytest.raises(ValueError):
        prediction_inversion_rate(["tiny", "large"], [])


# ---- observed_error -----------------------------------------------------------------------


def test_compute_per_image_error_all_correct_is_zero() -> None:
    pred = torch.zeros(2, 3, 3, dtype=torch.long)
    target = torch.zeros(2, 3, 3, dtype=torch.long)
    error = compute_per_image_error(pred, target)
    assert error.shape == (2,)
    assert torch.allclose(error, torch.zeros(2))


def test_compute_per_image_error_matches_known_fraction() -> None:
    pred = torch.zeros(1, 2, 2, dtype=torch.long)
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    target[0, 0, 0] = 1  # 1 of 4 pixels wrong
    error = compute_per_image_error(pred, target)
    assert float(error) == pytest.approx(0.25)


def test_compute_per_image_error_ignore_index_excluded() -> None:
    pred = torch.zeros(1, 2, 2, dtype=torch.long)
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    target[0, 0, 0] = 1  # 1 wrong pixel
    target[0, 0, 1] = IGNORE_INDEX  # excluded -- denominator shrinks to 3
    error = compute_per_image_error(pred, target)
    assert float(error) == pytest.approx(1 / 3)


def test_calibration_target_can_mask_labels_without_masking_deployment_feature() -> None:
    # The feature remains deployable (no target input), while the supervised error
    # label excludes invalid ground-truth pixels.
    logits = torch.zeros(1, 3, 2, 2)
    logits[:, 0] = 10.0
    target = torch.zeros(1, 2, 2, dtype=torch.long)
    target[0, 1, 1] = IGNORE_INDEX
    risk = compute_deployment_risk_score(logits)
    pred = logits.argmax(dim=1)
    error = compute_per_image_error(pred, target)
    assert risk.shape == error.shape == (1,)


# ---- calibrator serialization -----------------------------------------------------------------------


def test_risk_calibrator_round_trips_through_dict() -> None:
    calibrator = fit_risk_calibrator([0.1, 0.5, 0.9], [0.05, 0.2, 0.4], num_bins=3)
    restored = RiskCalibrator.from_dict(calibrator.to_dict())
    assert np.array_equal(calibrator.bin_edges, restored.bin_edges)
    assert np.array_equal(calibrator.bin_expected_error, restored.bin_expected_error)
    for score in (0.0, 0.3, 0.6, 1.0):
        assert calibrator.predict(score) == restored.predict(score)


def test_risk_calibrator_round_trips_through_json_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    calibrator = fit_risk_calibrator([0.1, 0.5, 0.9], [0.05, 0.2, 0.4], num_bins=3)
    path = tmp_path / "calibrator.json"
    calibrator.save(path)
    restored = RiskCalibrator.load(path)
    assert np.array_equal(calibrator.bin_edges, restored.bin_edges)
    assert np.array_equal(calibrator.bin_expected_error, restored.bin_expected_error)
