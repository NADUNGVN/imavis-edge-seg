import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from imavis_edge_seg.config import SupernetConfig  # noqa: E402
from imavis_edge_seg.models.blocks import SlimmableConv2d  # noqa: E402
from imavis_edge_seg.models.supernet import PaceSegSupernet  # noqa: E402
from imavis_edge_seg.training.quantization import (  # noqa: E402
    QATConv2d,
    QATSlimmableConv2d,
    apply_qat,
    fake_quantize_tensor,
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


def test_qatconv2d_preserves_state_dict_keys_and_values() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
    before_keys = list(model.state_dict().keys())
    before_state = {k: v.clone() for k, v in model.state_dict().items()}

    apply_qat(model)

    assert list(model.state_dict().keys()) == before_keys
    after_state = model.state_dict()
    assert all(torch.equal(before_state[k], after_state[k]) for k in before_keys)
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
    matters, not just "keys happen to match"."""
    fp32_model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
    fp32_state = fp32_model.state_dict()

    qat_model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1))
    apply_qat(qat_model)
    qat_model.load_state_dict(fp32_state)  # must not raise

    for k in fp32_state:
        assert torch.equal(qat_model.state_dict()[k], fp32_state[k])


# ---- QATSlimmableConv2d / supernet coverage ------------------------------------------


def test_qat_slimmable_conv2d_preserves_state_dict_keys_and_values() -> None:
    conv = SlimmableConv2d(3, 8, kernel_size=3, padding=1)
    before_keys = list(conv.state_dict().keys())
    before_state = {k: v.clone() for k, v in conv.state_dict().items()}

    q = QATSlimmableConv2d(conv)

    assert list(q.state_dict().keys()) == before_keys
    after_state = q.state_dict()
    assert all(torch.equal(before_state[k], after_state[k]) for k in before_keys)


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
    assert set(supernet.state_dict().keys()) == before_keys

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
