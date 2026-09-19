import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from imavis_edge_seg.config import SupernetConfig  # noqa: E402
from imavis_edge_seg.models.blocks import SlimmableConv2d  # noqa: E402
from imavis_edge_seg.models.supernet import PaceSegSupernet  # noqa: E402
from imavis_edge_seg.training.quantization import (  # noqa: E402
    QATConv2d,
    QATSlimmableConv2d,
    _percentile_via_topk,
    apply_qat,
    fake_quantize_tensor,
    fake_quantize_tensor_with_max,
    run_calibration,
)


def test_fake_quantize_tensor_zero_stays_zero() -> None:
    x = torch.zeros(4, requires_grad=True)
    out = fake_quantize_tensor(x)
    assert torch.equal(out, x)


def test_fake_quantize_tensor_bounded_error() -> None:
    torch.manual_seed(0)
    x = torch.randn(1000) * 3.0
    out = fake_quantize_tensor(x, num_bits=8)
    # INT8 symmetric quantization step is max_abs/127; per-element error must never
    # exceed half a quantization step (plus a little float slack).
    step = float(x.abs().max()) / 127.0
    assert torch.all((out - x).abs() <= step / 2 + 1e-4)


def test_fake_quantize_tensor_gradient_flows_straight_through() -> None:
    x = torch.randn(16, requires_grad=True)
    out = fake_quantize_tensor(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-3)


def test_fake_quantize_tensor_more_bits_means_less_error() -> None:
    torch.manual_seed(0)
    x = torch.randn(2000) * 5.0
    err_8bit = float((fake_quantize_tensor(x, num_bits=8) - x).abs().mean())
    err_4bit = float((fake_quantize_tensor(x, num_bits=4) - x).abs().mean())
    assert err_8bit < err_4bit


def test_qatconv2d_preserves_original_state_dict_keys_and_values() -> None:
    """apply_qat preserves every ORIGINAL key/value exactly (that's what lets an
    FP32 checkpoint reload -- see test_fp32_checkpoint_loads_into_qat_model_and_
    vice_versa below) and adds two NEW buffers per QAT layer (`calibrated_max` and
    `calibrated`, so a calibrated quantization range -- and the fact that it *is*
    calibrated -- both survive a checkpoint save/reload -- see
    test_run_calibration_survives_a_state_dict_round_trip)."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
    before_keys = set(model.state_dict().keys())
    before_state = {k: v.clone() for k, v in model.state_dict().items()}

    apply_qat(model)

    after_state = model.state_dict()
    assert before_keys <= set(after_state.keys())  # every original key still present
    assert all(torch.equal(before_state[k], after_state[k]) for k in before_keys)
    added_keys = set(after_state.keys()) - before_keys
    assert added_keys == {"0.calibrated_max", "0.calibrated", "2.calibrated_max", "2.calibrated"}
    assert isinstance(model[0], QATConv2d)
    assert isinstance(model[2], QATConv2d)


def test_qatconv2d_forward_shape_and_gradient_flow() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
    apply_qat(model)

    x = torch.randn(2, 3, 8, 8)
    y = model(x)
    assert y.shape == (2, 4, 8, 8)

    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_apply_qat_does_not_double_wrap_on_second_call() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    apply_qat(model)
    first_wrap = model[0]
    apply_qat(model)  # second call must not re-wrap an already-QATConv2d layer
    assert model[0] is first_wrap


def test_apply_qat_recurses_into_nested_modules() -> None:
    class Nested(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))

    model = Nested()
    apply_qat(model)
    assert isinstance(model.block[0], QATConv2d)


def test_fp32_checkpoint_loads_into_qat_model_and_vice_versa() -> None:
    """The RESEARCH_PLAN.md §5.2 workflow (FP32 teacher -> shared supernet -> QAT
    INT8) requires an FP32-trained state_dict to load cleanly into a QAT-converted
    model of the same architecture, and vice versa -- this is the property that
    matters, not just "keys happen to match". `strict=False` is required in the
    FP32->QAT direction specifically because the QAT model's `calibrated_max`/
    `calibrated` buffers (added for calibration support) have no counterpart in a
    plain FP32 state_dict -- they're new state, not renamed/dropped original
    state, so `strict=False`'s missing_keys is exactly those 4 new buffer keys and
    nothing else; a real key mismatch would show up as an unexpected_keys entry
    too, which this test also checks for."""
    fp32_model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
    fp32_state = fp32_model.state_dict()

    qat_model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
    apply_qat(qat_model)
    result = qat_model.load_state_dict(fp32_state, strict=False)
    assert set(result.missing_keys) == {
        "0.calibrated_max", "0.calibrated", "2.calibrated_max", "2.calibrated",
    }
    assert result.unexpected_keys == []

    for k in fp32_state:
        assert torch.equal(qat_model.state_dict()[k], fp32_state[k])


# ---- QATSlimmableConv2d / supernet coverage ------------------------------------------


def test_qat_slimmable_conv2d_preserves_original_state_dict_keys_and_values() -> None:
    conv = SlimmableConv2d(3, 8, kernel_size=3, padding=1)
    before_keys = set(conv.state_dict().keys())
    before_state = {k: v.clone() for k, v in conv.state_dict().items()}

    q = QATSlimmableConv2d(conv)

    after_state = q.state_dict()
    assert before_keys <= set(after_state.keys())
    assert all(torch.equal(before_state[k], after_state[k]) for k in before_keys)
    assert set(after_state.keys()) - before_keys == {"calibrated_max", "calibrated"}


def test_qat_slimmable_conv2d_forward_shape_and_gradient_flow() -> None:
    conv = SlimmableConv2d(8, 8, kernel_size=3, padding=1, depthwise=True)
    q = QATSlimmableConv2d(conv)

    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    y = q(x, active_in=4, active_out=4)
    assert y.shape == (2, 4, 8, 8)

    y.sum().backward()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()


def test_apply_qat_converts_slimmable_conv2d_in_a_real_supernet() -> None:
    """Verified end to end against the real PaceSegSupernet architecture (not just
    an isolated SlimmableConv2d) -- the FP32-vs-QAT-supernet gap this closes is the
    exact one `reports/qat_v1_20260913.md` flagged as follow-up work."""
    config = SupernetConfig()
    for level in config.levels:
        config.input_resolutions[level] = (64, 64)
    supernet = PaceSegSupernet(config)

    before_keys = set(supernet.state_dict().keys())
    apply_qat(supernet)
    after_keys = set(supernet.state_dict().keys())
    assert before_keys <= after_keys
    assert all(k.endswith(("calibrated_max", "calibrated")) for k in after_keys - before_keys)

    slimmable_count = sum(1 for m in supernet.modules() if isinstance(m, SlimmableConv2d))
    qat_slimmable_count = sum(1 for m in supernet.modules() if isinstance(m, QATSlimmableConv2d))
    assert slimmable_count == qat_slimmable_count > 0  # every SlimmableConv2d converted

    x = torch.randn(1, 3, 64, 64)
    for level in config.levels:
        logits = supernet(x, level)
        assert torch.isfinite(logits).all()


def test_apply_qat_does_not_double_wrap_slimmable_conv2d() -> None:
    conv = SlimmableConv2d(3, 4, kernel_size=3, padding=1)
    model = nn.Module()
    model.conv = conv  # type: ignore[assignment]
    apply_qat(model)
    first_wrap = model.conv
    apply_qat(model)
    assert model.conv is first_wrap


# ---- calibrated (not dynamic) quantization ranges ------------------------------------


def test_fake_quantize_tensor_with_max_uses_given_max_not_input_own_max() -> None:
    x = torch.tensor([10.0, -10.0, 0.5])
    out = fake_quantize_tensor_with_max(x, max_abs=100.0, num_bits=8)
    # Quantization step is 100/127 here, not 10/127 -- a much coarser grid than
    # fake_quantize_tensor(x) would use, since that would derive scale from x itself.
    step = 100.0 / 127.0
    assert torch.all((out - x).abs() <= step / 2 + 1e-4)


def test_fake_quantize_tensor_with_max_zero_returns_input_unchanged() -> None:
    x = torch.randn(4)
    assert torch.equal(fake_quantize_tensor_with_max(x, max_abs=0.0), x)


def test_run_calibration_raises_if_model_has_no_qat_layers() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))  # apply_qat never called
    with pytest.raises(ValueError):
        run_calibration(model, [lambda: model(torch.randn(1, 3, 8, 8))])


def test_run_calibration_sets_calibrated_true_and_freezes_a_max() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    apply_qat(model)
    layer = model[0]
    assert not layer.calibrated  # type: ignore[union-attr]

    calls = [lambda v=v: model(torch.full((1, 3, 8, 8), v)) for v in (1.0, -5.0, 2.0)]
    run_calibration(model, calls)

    assert layer.calibrated  # type: ignore[union-attr]
    assert not layer.calibrating  # type: ignore[union-attr]
    assert float(layer.calibrated_max) == pytest.approx(5.0)  # the largest |value| seen


def test_run_calibration_frozen_scale_used_on_later_forward_calls_not_dynamic() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False))
    apply_qat(model)
    # Calibrate on inputs no bigger than magnitude 5 ...
    run_calibration(model, [lambda: model(torch.full((1, 3, 8, 8), 5.0))])
    layer = model[0]
    assert float(layer.calibrated_max) == pytest.approx(5.0)  # type: ignore[arg-type]

    # ... then a much larger later input must still be quantized against the frozen
    # max=5 grid (visible as heavy clipping/coarse rounding), not its own (dynamic)
    # max=50 -- i.e. calibration mode must not silently revert to dynamic behavior.
    with torch.no_grad():
        big_input = torch.full((1, 3, 8, 8), 50.0)
        quantized_weight = fake_quantize_tensor(layer.weight, layer.num_bits)  # type: ignore[union-attr]
        expected_with_frozen_scale = fake_quantize_tensor_with_max(big_input, 5.0, layer.num_bits)  # type: ignore[union-attr]
        import torch.nn.functional as F

        expected_output = F.conv2d(expected_with_frozen_scale, quantized_weight, padding=layer.padding)  # type: ignore[union-attr]
        actual_output = layer(big_input)
    assert torch.allclose(actual_output, expected_output)


def test_run_calibration_max_grows_monotonically_across_calls_never_shrinks() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    apply_qat(model)
    run_calibration(
        model,
        [
            lambda: model(torch.full((1, 3, 8, 8), 10.0)),
            lambda: model(torch.full((1, 3, 8, 8), 2.0)),  # smaller -- must not shrink the max
            lambda: model(torch.full((1, 3, 8, 8), 7.0)),
        ],
    )
    assert float(model[0].calibrated_max) == pytest.approx(10.0)  # type: ignore[arg-type]


def test_run_calibration_survives_a_state_dict_round_trip() -> None:
    """Both `calibrated_max` and the `calibrated` flag itself are buffers, so a
    freshly-constructed (apply_qat'd, never-calibrated) model that loads a
    calibrated checkpoint's state_dict resumes in calibrated mode automatically --
    a caller does NOT need to remember to call run_calibration again or flip
    `calibrated` manually just to *evaluate* an already-calibrated checkpoint.
    `calibrating` (transient, only meaningful mid-calibration) is the one flag that
    is deliberately not a buffer and does not need to survive."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    apply_qat(model)
    run_calibration(model, [lambda: model(torch.full((1, 3, 8, 8), 3.0))])
    saved_state = model.state_dict()

    reloaded = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    apply_qat(reloaded)
    assert not bool(reloaded[0].calibrated)  # type: ignore[union-attr]
    reloaded.load_state_dict(saved_state)

    assert float(reloaded[0].calibrated_max) == pytest.approx(3.0)
    assert bool(reloaded[0].calibrated)  # type: ignore[union-attr]
    assert not reloaded[0].calibrating  # type: ignore[union-attr]


def test_run_calibration_covers_every_qat_layer_in_a_real_supernet() -> None:
    config = SupernetConfig()
    for level in config.levels:
        config.input_resolutions[level] = (64, 64)
    supernet = PaceSegSupernet(config)
    apply_qat(supernet)

    x = torch.randn(1, 3, 64, 64)
    run_calibration(supernet, [lambda level=level: supernet(x, level) for level in config.levels])

    qat_layers = [m for m in supernet.modules() if isinstance(m, QATSlimmableConv2d)]
    assert qat_layers
    assert all(m.calibrated for m in qat_layers)
    assert all(float(m.calibrated_max) > 0.0 for m in qat_layers)


# ---- ema_percentile observer (2026-09-20, QAT-rescue 2x2 screen) ---------------------


def test_percentile_via_topk_matches_known_value() -> None:
    # 1000 values 1..1000 -- the 99.9th percentile should be very close to 999.
    x = torch.arange(1, 1001, dtype=torch.float32)
    result = _percentile_via_topk(x, percentile=0.999)
    assert float(result) == pytest.approx(999.0, abs=1.0)


def test_percentile_via_topk_ignores_a_single_extreme_outlier() -> None:
    # 2000 values -- the 99.9th percentile excludes the top 2 (round(0.001*2000)=2),
    # so with exactly one outlier, the 2nd-largest ("normal") value is returned, not
    # the outlier itself.
    x = torch.cat([torch.ones(1999), torch.tensor([1e6])])
    p999 = _percentile_via_topk(x, percentile=0.999)
    p_max = x.abs().max()
    assert float(p999) < float(p_max)  # the outlier itself is excluded from the 99.9th percentile
    assert float(p999) == pytest.approx(1.0, abs=0.1)


def test_run_calibration_ema_percentile_ignores_a_single_extreme_outlier_batch() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    apply_qat(model)
    normal_batches = [lambda: model(torch.full((1, 3, 8, 8), 2.0)) for _ in range(5)]
    outlier_batch = [lambda: model(torch.full((1, 3, 8, 8), 1000.0))]
    run_calibration(model, normal_batches + outlier_batch, observer="ema_percentile", momentum=0.9)
    # A hard running max would jump straight to ~1000; the EMA should stay far below it,
    # since the outlier batch is downweighted by (1 - momentum) rather than taking over.
    assert float(model[0].calibrated_max) < 500.0


def test_run_calibration_max_observer_is_unaffected_by_the_new_ema_percentile_path() -> None:
    """Default observer="max" must behave exactly as before -- a regression check
    that adding ema_percentile did not change the original behavior."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    apply_qat(model)
    run_calibration(model, [lambda v=v: model(torch.full((1, 3, 8, 8), v)) for v in (1.0, 5.0, 2.0)])
    assert float(model[0].calibrated_max) == pytest.approx(5.0)


def test_run_calibration_ema_percentile_covers_every_qat_layer_in_a_real_supernet() -> None:
    config = SupernetConfig()
    for level in config.levels:
        config.input_resolutions[level] = (64, 64)
    supernet = PaceSegSupernet(config)
    apply_qat(supernet)

    x = torch.randn(1, 3, 64, 64)
    run_calibration(
        supernet,
        [lambda level=level: supernet(x, level) for level in config.levels],
        observer="ema_percentile",
    )

    qat_layers = [m for m in supernet.modules() if isinstance(m, QATSlimmableConv2d)]
    assert qat_layers
    assert all(m.calibrated for m in qat_layers)
    assert all(float(m.calibrated_max) > 0.0 for m in qat_layers)
