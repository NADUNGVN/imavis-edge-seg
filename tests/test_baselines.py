import io
from pathlib import Path

import pytest
from rich.console import Console

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from imavis_edge_seg.config import ExperimentConfig  # noqa: E402
from imavis_edge_seg.models import PaceSegSupernet, extract_subnet  # noqa: E402
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


_IMPLEMENTED = {"mobilenetv3_deeplabv3", "fast_scnn", "bisenetv2", "ddrnet23_slim", "segformer_b0", "pace_large"}


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


def test_run_baseline_training_model_override_skips_build_baseline_model(tmp_path: Path) -> None:
    # scripts/train_exported_subnet.py's whole reason for the `model=` parameter: hand
    # in an already-built module (e.g. models.subnet.extract_subnet's output) and have
    # it trained in place, without run_baseline_training building `baseline_name` itself.
    config = _tiny_config()
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    given_model = build_baseline_model("fast_scnn", num_classes=config.supernet.num_classes)
    torch.manual_seed(config.seed)
    before_state = {k: v.clone() for k, v in given_model.state_dict().items()}

    trained = run_baseline_training(
        "not_a_real_baseline_name",
        config,
        dataloader,
        output_dir=tmp_path,
        device="cpu",
        model=given_model,
    )

    assert trained is given_model
    changed = any(not torch.equal(before_state[k], v) for k, v in trained.state_dict().items())
    assert changed, "weights should change after training steps"


def test_run_baseline_training_calibrate_fits_and_freezes_real_calibration_data(tmp_path: Path) -> None:
    """Mirrors test_training.py's identical supernet-side test: --calibrate builds a
    real calibration set from config.datasets and every QAT-converted layer ends up
    `calibrated=True` with a nonzero observed max -- the wiring
    scripts/train_baseline.py --calibrate and scripts/train_exported_subnet.py
    --calibrate both depend on."""
    import numpy as np

    from imavis_edge_seg.config import DatasetConfig

    cityscapes_root = tmp_path / "cityscapes"
    rgb = (np.random.rand(*_TEST_HW, 3) * 255).astype("uint8")
    label_ids = np.full(_TEST_HW, 7, dtype="uint8")
    for i in range(4):
        img_path = cityscapes_root / f"leftImg8bit/train/city/city_{i:06d}_000019_leftImg8bit.png"
        lbl_path = cityscapes_root / f"gtFine/train/city/city_{i:06d}_000019_gtFine_labelIds.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(img_path)
        Image.fromarray(label_ids).save(lbl_path)

    config = _tiny_config(max_steps=1, checkpoint_interval_steps=1)
    config.datasets = [DatasetConfig(name="cityscapes", root=cityscapes_root, split="train")]
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    trained = run_baseline_training(
        "fast_scnn", config, dataloader, output_dir=tmp_path / "run", device="cpu",
        qat=True, calibrate=True, calibration_images=4,
    )

    qat_layers = [m for m in trained.modules() if isinstance(m, QATConv2d)]
    assert qat_layers
    assert all(bool(m.calibrated) for m in qat_layers)
    assert all(float(m.calibrated_max) > 0.0 for m in qat_layers)


def test_run_baseline_training_calibrate_without_qat_raises() -> None:
    config = _tiny_config()
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)
    with pytest.raises(ValueError, match="calibrate"):
        run_baseline_training(
            "fast_scnn", config, dataloader, output_dir=Path("unused"), device="cpu", calibrate=True
        )


def test_run_baseline_training_on_extracted_subnet_end_to_end(tmp_path: Path) -> None:
    """The exported-subnet QAT-rescue path (scripts/train_exported_subnet.py, QAT 2x2
    screen cells 3-4, docs/COORDINATION_LOG.md open thread #1): extract_subnet's
    output -- despite being an independent nn.Conv2d graph, not fast_scnn/etc. --
    trains through run_baseline_training exactly like any other baseline model, and
    QAT converts every one of its conv layers to QATConv2d (never QATSlimmableConv2d,
    confirming extract_subnet really did produce plain, independent layers)."""
    config = _tiny_config()
    for level in config.supernet.levels:
        config.supernet.input_resolutions[level] = _TEST_HW
    supernet = PaceSegSupernet(config.supernet)

    subnet = extract_subnet(supernet, "small")  # type: ignore[arg-type]
    assert not subnet.training, "extract_subnet's output starts in eval() mode"
    subnet.train()

    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)
    before_state = {k: v.clone() for k, v in subnet.state_dict().items()}

    trained = run_baseline_training(
        "exported_subnet_small", config, dataloader, output_dir=tmp_path, device="cpu",
        qat=True, model=subnet,
    )

    assert trained is subnet
    assert any(isinstance(m, QATConv2d) for m in trained.modules())
    from imavis_edge_seg.training.quantization import QATSlimmableConv2d

    assert not any(isinstance(m, QATSlimmableConv2d) for m in trained.modules())
    changed = any(not torch.equal(before_state[k], v) for k, v in trained.state_dict().items())
    assert changed, "weights should change after training steps"


def test_exported_subnet_qat_checkpoint_reloads_into_a_fresh_skeleton(tmp_path: Path) -> None:
    """scripts/evaluate_baseline.py --exported-subnet-level rebuilds the architecture
    as `apply_qat(extract_subnet(PaceSegSupernet(config.supernet), level))` from a
    freshly-constructed (differently-weighted) supernet, then load_state_dict's a
    checkpoint saved by scripts/train_exported_subnet.py's training path. This is the
    exact state_dict round trip that path depends on -- covers the gap flagged in
    reports/qat_rescue_2x2_screen_infra_20260920.md."""
    from imavis_edge_seg.training.quantization import apply_qat

    config = _tiny_config()
    for level in config.supernet.levels:
        config.supernet.input_resolutions[level] = _TEST_HW

    train_supernet = PaceSegSupernet(config.supernet)
    subnet = extract_subnet(train_supernet, "small")  # type: ignore[arg-type]
    subnet.train()
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)
    trained = run_baseline_training(
        "exported_subnet_small", config, dataloader, output_dir=tmp_path, device="cpu",
        qat=True, model=subnet,
    )
    checkpoint_path = sorted((tmp_path / "checkpoints").glob("*.pt"))[-1]

    # A *different* random supernet init -- evaluate_baseline.py never sees the
    # trained weights before load_state_dict, only the architecture.
    fresh_supernet = PaceSegSupernet(config.supernet)
    skeleton = extract_subnet(fresh_supernet, "small")  # type: ignore[arg-type]
    skeleton = apply_qat(skeleton)  # type: ignore[assignment]

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    skeleton.load_state_dict(checkpoint["model_state_dict"])  # must not raise

    trained_state = trained.state_dict()
    for key, value in skeleton.state_dict().items():
        assert torch.equal(value, trained_state[key])


def test_run_baseline_training_pace_large_updates_batchnorm_running_stats(tmp_path: Path) -> None:
    """StaticPaceSegSubnet starts in eval mode; training must still update BN stats."""
    config = _tiny_config()
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)
    trained = run_baseline_training("pace_large", config, dataloader, output_dir=tmp_path, device="cpu")
    bns = [m for m in trained.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns
    assert all(int(m.num_batches_tracked) > 0 for m in bns)
