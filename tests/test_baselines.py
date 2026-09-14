import io
from pathlib import Path

import pytest
from rich.console import Console

torch = pytest.importorskip("torch")

from imavis_edge_seg.config import ExperimentConfig  # noqa: E402
from imavis_edge_seg.models.baselines import BASELINE_NAMES, build_baseline_model  # noqa: E402
from imavis_edge_seg.training.baseline_trainer import run_baseline_training  # noqa: E402
from imavis_edge_seg.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from imavis_edge_seg.training.quantization import QATConv2d  # noqa: E402

_TEST_HW = (64, 64)


def test_build_baseline_model_mobilenetv3_deeplabv3_output_shape() -> None:
    model = build_baseline_model("mobilenetv3_deeplabv3", num_classes=19)
    image = torch.randn(2, 3, *_TEST_HW)
    logits = model(image)
    assert logits.shape == (2, 19, *_TEST_HW)


@pytest.mark.parametrize("name", ["fast_scnn", "bisenetv2", "ddrnet23_slim", "segformer_b0"])
def test_build_baseline_model_from_scratch_archs_output_shape_and_gradient_flow(name: str) -> None:
    model = build_baseline_model(name, num_classes=19)
    image = torch.randn(2, 3, *_TEST_HW, requires_grad=True)
    logits = model(image)
    assert logits.shape == (2, 19, *_TEST_HW)
    logits.sum().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert all(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in model.parameters()
        if param.requires_grad
    ), "every parameter should receive a finite gradient -- no dead/detached path"


_IMPLEMENTED = {"mobilenetv3_deeplabv3", "fast_scnn", "bisenetv2", "ddrnet23_slim", "segformer_b0"}


@pytest.mark.parametrize("name", [n for n in BASELINE_NAMES if n not in _IMPLEMENTED])
def test_build_baseline_model_unimplemented_raises_not_config_error(name: str) -> None:
    # These are required baselines (RESEARCH_PLAN.md §7) not yet implemented -- the
    # factory must say so explicitly, not raise a generic/misleading error or silently
    # return something wrong.
    with pytest.raises(NotImplementedError):
        build_baseline_model(name)


def test_build_baseline_model_unknown_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        build_baseline_model("not_a_real_baseline")


def _tiny_config(**overrides: object) -> ExperimentConfig:
    config = ExperimentConfig(experiment_id="test-baseline")
    for level in config.supernet.levels:
        config.supernet.input_resolutions[level] = _TEST_HW
    config.training.max_steps = 2
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.log_interval_steps = 1
    config.training.checkpoint_interval_steps = 2
    for key, value in overrides.items():
        setattr(config.training, key, value)
    return config


class _TinySegDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.randn(3, *_TEST_HW)
        mask = torch.randint(0, 19, _TEST_HW)
        return image, mask


def test_run_baseline_training_end_to_end_updates_weights_and_checkpoints(tmp_path: Path) -> None:
    config = _tiny_config()
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    before = build_baseline_model("mobilenetv3_deeplabv3", num_classes=config.supernet.num_classes)
    torch.manual_seed(config.seed)
    before_state = {k: v.clone() for k, v in before.state_dict().items()}

    trained = run_baseline_training(
        "mobilenetv3_deeplabv3", config, dataloader, output_dir=tmp_path, device="cpu"
    )

    changed = any(not torch.equal(before_state[k], v) for k, v in trained.state_dict().items())
    assert changed, "weights should change after training steps"

    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) >= 1

    checkpoint = load_checkpoint(checkpoints[0])
    assert set(checkpoint["model_state_dict"].keys()) == set(trained.state_dict().keys())


def test_run_baseline_training_resumes_from_existing_checkpoint(tmp_path: Path) -> None:
    from torch.optim import AdamW

    config = _tiny_config(max_steps=4, checkpoint_interval_steps=2)
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    model = build_baseline_model("mobilenetv3_deeplabv3", num_classes=config.supernet.num_classes)
    optimizer = AdamW(model.parameters(), lr=config.training.lr)
    save_checkpoint(tmp_path / "checkpoints" / "step_00000002.pt", model, optimizer, step=2, config=config)

    buffer = io.StringIO()
    console = Console(file=buffer, width=200)
    run_baseline_training(
        "mobilenetv3_deeplabv3", config, dataloader, output_dir=tmp_path, console=console, device="cpu"
    )

    log = buffer.getvalue()
    assert "resumed from" in log and "step 2" in log
    checkpoints = sorted((tmp_path / "checkpoints").glob("*.pt"))
    assert [c.name for c in checkpoints] == ["step_00000002.pt", "step_00000004.pt"]
    assert load_checkpoint(checkpoints[-1])["step"] == 4


def test_run_baseline_training_qat_converts_conv_layers_and_trains(tmp_path: Path) -> None:
    config = _tiny_config()
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    trained = run_baseline_training(
        "fast_scnn", config, dataloader, output_dir=tmp_path, device="cpu", qat=True
    )

    assert any(isinstance(m, QATConv2d) for m in trained.modules())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in trained.parameters()
    ), "QAT training should still produce finite gradients on every parameter"


def test_run_baseline_training_init_checkpoint_starts_from_fp32_weights(tmp_path: Path) -> None:
    from torch.optim import AdamW

    config = _tiny_config()
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    fp32_model = build_baseline_model("fast_scnn", num_classes=config.supernet.num_classes)
    fp32_optimizer = AdamW(fp32_model.parameters(), lr=config.training.lr)
    fp32_ckpt = tmp_path / "fp32_step_00000010.pt"
    save_checkpoint(fp32_ckpt, fp32_model, fp32_optimizer, step=10, config=config)
    fp32_weight = fp32_model.downsample[0][0].weight.clone()

    qat_output_dir = tmp_path / "qat_run"
    trained = run_baseline_training(
        "fast_scnn",
        config,
        dataloader,
        output_dir=qat_output_dir,
        device="cpu",
        qat=True,
        init_checkpoint=fp32_ckpt,
    )

    # A QAT run initialized from the FP32 checkpoint starts from those weights (then
    # trains further, so it won't be bit-identical, but should be close after just a
    # couple of steps) rather than a fresh random init.
    trained_weight = trained.downsample[0][0].weight
    assert torch.allclose(trained_weight, fp32_weight, atol=0.1)
    assert isinstance(trained.downsample[0][0], QATConv2d)
