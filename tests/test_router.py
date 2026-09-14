import numpy as np
import pytest

torch = pytest.importorskip("torch")

from imavis_edge_seg.config import RouterConfig  # noqa: E402
from imavis_edge_seg.data.labels import IGNORE_INDEX  # noqa: E402
from imavis_edge_seg.router.calibrator import fit_risk_calibrator  # noqa: E402
from imavis_edge_seg.router.policy import select_level  # noqa: E402
from imavis_edge_seg.router.risk_probe import compute_risk_score  # noqa: E402
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
