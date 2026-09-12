import pytest

torch = pytest.importorskip("torch")

from imavis_edge_seg.data.labels import IGNORE_INDEX  # noqa: E402
from imavis_edge_seg.evaluation.calibration import CalibrationAccumulator  # noqa: E402

_NUM_CLASSES = 3


def _confident_logits(target: "torch.Tensor", correct: bool, margin: float = 20.0) -> "torch.Tensor":
    """(B, C, H, W) logits that assign near-1.0 softmax probability to `target`'s
    class if `correct`, else to a different (wrong) class, everywhere target !=
    IGNORE_INDEX."""
    shape = (target.shape[0], _NUM_CLASSES, *target.shape[1:])
    logits = torch.zeros(shape)
    predicted_class = target if correct else (target + 1) % _NUM_CLASSES
    for c in range(_NUM_CLASSES):
        mask = predicted_class == c
        logits[:, c][mask] = margin
    return logits


def test_confidently_correct_gives_near_zero_ece_nll_brier_and_aurc() -> None:
    target = torch.randint(0, _NUM_CLASSES, (2, 8, 8))
    logits = _confident_logits(target, correct=True)

    acc = CalibrationAccumulator(num_classes=_NUM_CLASSES, num_bins=10)
    acc.update(logits, target)
    result = acc.compute()

    assert result.ece == pytest.approx(0.0, abs=1e-3)
    assert result.nll == pytest.approx(0.0, abs=1e-3)
    assert result.brier == pytest.approx(0.0, abs=1e-3)
    assert result.aurc == pytest.approx(0.0, abs=1e-3)
    assert result.num_pixels == 2 * 8 * 8


def test_confidently_wrong_gives_high_ece_nll_brier_and_aurc() -> None:
    target = torch.randint(0, _NUM_CLASSES, (2, 8, 8))
    logits = _confident_logits(target, correct=False)

    acc = CalibrationAccumulator(num_classes=_NUM_CLASSES, num_bins=10)
    acc.update(logits, target)
    result = acc.compute()

    assert result.ece == pytest.approx(1.0, abs=1e-2)  # confidence ~1, accuracy 0
    assert result.nll > 10.0  # -log(~0) is large
    assert result.brier == pytest.approx(2.0, abs=1e-2)  # max Brier for one-hot
    assert result.aurc == pytest.approx(1.0, abs=1e-2)  # every pixel wrong at every coverage


def test_ignore_index_pixels_excluded_from_every_metric() -> None:
    target_clean = torch.randint(0, _NUM_CLASSES, (1, 4, 4))
    logits_clean = _confident_logits(target_clean, correct=True)

    # Same clean pixels, plus a block of ignored pixels with confidently *wrong*
    # logits -- if excluded correctly, results must match the clean-only run exactly.
    target_with_ignore = torch.cat([target_clean, torch.full((1, 4, 4), IGNORE_INDEX)], dim=1)
    wrong_block = _confident_logits(torch.randint(0, _NUM_CLASSES, (1, 4, 4)), correct=False)
    logits_with_ignore = torch.cat([logits_clean, wrong_block], dim=2)

    acc_clean = CalibrationAccumulator(num_classes=_NUM_CLASSES, num_bins=10)
    acc_clean.update(logits_clean, target_clean)
    clean_result = acc_clean.compute()

    acc_with_ignore = CalibrationAccumulator(num_classes=_NUM_CLASSES, num_bins=10)
    acc_with_ignore.update(logits_with_ignore, target_with_ignore)
    ignore_result = acc_with_ignore.compute()

    assert ignore_result.num_pixels == clean_result.num_pixels
    assert ignore_result.ece == pytest.approx(clean_result.ece, abs=1e-6)
    assert ignore_result.nll == pytest.approx(clean_result.nll, abs=1e-6)
    assert ignore_result.brier == pytest.approx(clean_result.brier, abs=1e-6)


def test_update_is_streaming_consistent_across_multiple_batches() -> None:
    torch.manual_seed(0)
    target1 = torch.randint(0, _NUM_CLASSES, (1, 6, 6))
    target2 = torch.randint(0, _NUM_CLASSES, (1, 6, 6))
    logits1 = torch.randn(1, _NUM_CLASSES, 6, 6)
    logits2 = torch.randn(1, _NUM_CLASSES, 6, 6)

    streaming = CalibrationAccumulator(num_classes=_NUM_CLASSES, num_bins=10)
    streaming.update(logits1, target1)
    streaming.update(logits2, target2)
    streaming_result = streaming.compute()

    batched = CalibrationAccumulator(num_classes=_NUM_CLASSES, num_bins=10)
    batched.update(torch.cat([logits1, logits2], dim=0), torch.cat([target1, target2], dim=0))
    batched_result = batched.compute()

    assert streaming_result.ece == pytest.approx(batched_result.ece, abs=1e-9)
    assert streaming_result.nll == pytest.approx(batched_result.nll, abs=1e-9)
    assert streaming_result.brier == pytest.approx(batched_result.brier, abs=1e-9)
    assert streaming_result.aurc == pytest.approx(batched_result.aurc, abs=1e-9)
    assert streaming_result.num_pixels == batched_result.num_pixels


def test_aurc_lower_when_confidence_actually_tracks_correctness() -> None:
    # Half the pixels: high-confidence and correct. Other half: low-confidence and
    # wrong. A good risk-coverage curve should reject the low-confidence half first,
    # giving near-zero risk at up to 50% coverage -- much lower AURC than a run where
    # confidence carries no information about correctness.
    target = torch.randint(0, _NUM_CLASSES, (1, 4, 8))
    good_half = _confident_logits(target[:, :, :4], correct=True, margin=20.0)
    bad_half = _confident_logits(target[:, :, 4:], correct=False, margin=1.0)
    logits = torch.cat([good_half, bad_half], dim=3)

    informative = CalibrationAccumulator(num_classes=_NUM_CLASSES, num_bins=20)
    informative.update(logits, target)
    informative_result = informative.compute()

    # Swap which half is confident vs. correct -- same overall accuracy (50%), but
    # confidence now anti-correlates with correctness -- AURC must be worse (higher).
    bad_half_confident = _confident_logits(target[:, :, :4], correct=False, margin=20.0)
    good_half_unconfident = _confident_logits(target[:, :, 4:], correct=True, margin=1.0)
    anti_logits = torch.cat([bad_half_confident, good_half_unconfident], dim=3)
    anti = CalibrationAccumulator(num_classes=_NUM_CLASSES, num_bins=20)
    anti.update(anti_logits, target)
    anti_result = anti.compute()

    assert informative_result.aurc < anti_result.aurc
