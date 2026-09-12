import io
import random
from pathlib import Path

import pytest
from rich.console import Console

torch = pytest.importorskip("torch")

from imavis_edge_seg.config import ExperimentConfig  # noqa: E402
from imavis_edge_seg.models.supernet import PaceSegSupernet  # noqa: E402
from imavis_edge_seg.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from imavis_edge_seg.training.losses import (  # noqa: E402
    boundary_aware_segmentation_loss,
    boundary_weight_map,
    distillation_kl_loss,
    resize_image,
    resize_mask,
)
from imavis_edge_seg.training.sandwich import sample_training_levels  # noqa: E402
from imavis_edge_seg.training.step import train_step  # noqa: E402
from imavis_edge_seg.training.trainer import run_training  # noqa: E402

_TEST_HW = (64, 64)


def _tiny_config(**overrides: object) -> ExperimentConfig:
    config = ExperimentConfig(experiment_id="test-training")
    for level in config.supernet.levels:
        config.supernet.input_resolutions[level] = _TEST_HW
    config.training.max_steps = 2
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.training.log_interval_steps = 1
    config.training.checkpoint_interval_steps = 2
    config.training.sandwich_num_random_middle = 1
    for key, value in overrides.items():
        setattr(config.training, key, value)
    return config


# ---- sandwich rule ----------------------------------------------------------------


def test_sandwich_always_includes_smallest_and_largest() -> None:
    config = _tiny_config()
    rng = random.Random(0)
    for _ in range(20):
        levels = sample_training_levels(config.supernet, config.training, rng)
        assert config.supernet.levels[0] in levels
        assert config.supernet.levels[-1] in levels
        assert levels[0] == config.supernet.levels[-1]  # teacher first


def test_sandwich_respects_random_middle_count() -> None:
    config = _tiny_config(sandwich_num_random_middle=2)
    rng = random.Random(1)
    levels = sample_training_levels(config.supernet, config.training, rng)
    # tiny, small, medium, large -> teacher(large) + up to 2 middle + smallest(tiny)
    assert len(levels) <= 4
    assert len(levels) >= 3  # at least teacher + smallest + 1 middle in a 4-level space


# ---- losses -------------------------------------------------------------------------


def test_boundary_weight_map_flags_edges_and_zeros_ignored() -> None:
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 2:, :] = 1  # horizontal edge between rows 1 and 2
    target[0, 0, 0] = 255  # ignored pixel
    weights = boundary_weight_map(target)
    assert weights[0, 0, 0] == 0.0  # ignored -> zero weight
    assert weights[0, 1, 1] == 3.0  # adjacent to the edge -> boosted
    assert weights[0, 3, 3] == 1.0  # far from edge, valid -> baseline


def test_boundary_aware_segmentation_loss_is_finite() -> None:
    logits = torch.randn(2, 19, 8, 8, requires_grad=True)
    target = torch.randint(0, 19, (2, 8, 8))
    loss = boundary_aware_segmentation_loss(logits, target, boundary_weight=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_distillation_kl_loss_zero_when_student_equals_teacher() -> None:
    logits = torch.randn(2, 19, 8, 8)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    loss = distillation_kl_loss(logits, logits.clone(), valid)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-5)


def test_distillation_kl_loss_positive_when_different() -> None:
    student = torch.randn(2, 19, 8, 8)
    teacher = torch.randn(2, 19, 8, 8)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    loss = distillation_kl_loss(student, teacher, valid)
    assert loss > 0


def test_resize_image_and_mask_change_shape() -> None:
    image = torch.randn(1, 3, 8, 8)
    mask = torch.randint(0, 19, (1, 8, 8))
    resized_image = resize_image(image, (16, 16))
    resized_mask = resize_mask(mask, (16, 16))
    assert resized_image.shape == (1, 3, 16, 16)
    assert resized_mask.shape == (1, 16, 16)


# ---- train_step ---------------------------------------------------------------------


def test_train_step_produces_finite_loss_and_gradients() -> None:
    config = _tiny_config()
    supernet = PaceSegSupernet(config.supernet)
    image = torch.randn(2, 3, *_TEST_HW)
    mask = torch.randint(0, 19, (2, *_TEST_HW))
    rng = random.Random(0)

    result = train_step(supernet, image, mask, config, rng)
    assert torch.isfinite(result.total_loss)
    assert config.supernet.levels[-1] in result.levels_trained

    result.total_loss.backward()
    grad_norms = [p.grad.norm().item() for p in supernet.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(g > 0 for g in grad_norms)


def test_train_step_distills_from_teacher_for_smaller_levels() -> None:
    config = _tiny_config(sandwich_num_random_middle=0)
    supernet = PaceSegSupernet(config.supernet)
    image = torch.randn(2, 3, *_TEST_HW)
    mask = torch.randint(0, 19, (2, *_TEST_HW))
    rng = random.Random(0)

    result = train_step(supernet, image, mask, config, rng)
    smallest = config.supernet.levels[0]
    largest = config.supernet.levels[-1]
    if smallest != largest:
        assert smallest in result.distill_losses
    assert largest not in result.distill_losses  # teacher has no distillation term


# ---- checkpoint ---------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = _tiny_config()
    supernet = PaceSegSupernet(config.supernet)
    optimizer = torch.optim.AdamW(supernet.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "step_00000001.pt"

    save_checkpoint(ckpt_path, supernet, optimizer, step=1, config=config)
    assert ckpt_path.exists()

    checkpoint = load_checkpoint(ckpt_path)
    assert checkpoint["step"] == 1
    assert checkpoint["config_hash"] == config.config_hash()
    assert set(checkpoint["model_state_dict"].keys()) == set(supernet.state_dict().keys())


# ---- end-to-end (tiny synthetic run) -------------------------------------------------


class _TinySegDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.randn(3, *_TEST_HW)
        mask = torch.randint(0, 19, _TEST_HW)
        return image, mask


def test_run_training_end_to_end_updates_weights_and_checkpoints(tmp_path: Path) -> None:
    config = _tiny_config()
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    before = PaceSegSupernet(config.supernet)
    torch.manual_seed(config.seed)
    before_state = {k: v.clone() for k, v in before.state_dict().items()}

    trained = run_training(config, dataloader, output_dir=tmp_path, device="cpu")

    changed = any(
        not torch.equal(before_state[k], v) for k, v in trained.state_dict().items()
    )
    assert changed, "weights should change after training steps"

    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) >= 1


# ---- resume from checkpoint ---------------------------------------------------------


def test_run_training_resumes_from_existing_checkpoint(tmp_path: Path) -> None:
    from torch.optim import AdamW

    config = _tiny_config(max_steps=4, checkpoint_interval_steps=2)
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    # Simulate a run that already reached step 2 (e.g. before an accidental kill):
    # a real step_00000002.pt checkpoint, same config, pre-existing in output_dir.
    supernet = PaceSegSupernet(config.supernet)
    optimizer = AdamW(supernet.parameters(), lr=config.training.lr)
    save_checkpoint(tmp_path / "checkpoints" / "step_00000002.pt", supernet, optimizer, step=2, config=config)

    buffer = io.StringIO()
    console = Console(file=buffer, width=200)
    run_training(config, dataloader, output_dir=tmp_path, console=console, device="cpu")

    log = buffer.getvalue()
    assert "resumed from" in log and "step 2" in log

    checkpoints = sorted((tmp_path / "checkpoints").glob("*.pt"))
    assert [c.name for c in checkpoints] == ["step_00000002.pt", "step_00000004.pt"]
    assert load_checkpoint(checkpoints[-1])["step"] == 4


def test_run_training_resume_already_at_max_steps_does_nothing(tmp_path: Path) -> None:
    from torch.optim import AdamW

    config = _tiny_config(max_steps=2, checkpoint_interval_steps=2)
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    supernet = PaceSegSupernet(config.supernet)
    optimizer = AdamW(supernet.parameters(), lr=config.training.lr)
    save_checkpoint(tmp_path / "checkpoints" / "step_00000002.pt", supernet, optimizer, step=2, config=config)
    saved_state = {k: v.clone() for k, v in supernet.state_dict().items()}

    buffer = io.StringIO()
    console = Console(file=buffer, width=200)
    returned = run_training(config, dataloader, output_dir=tmp_path, console=console, device="cpu")

    assert "nothing to do" in buffer.getvalue()
    assert all(torch.equal(saved_state[k], v) for k, v in returned.state_dict().items())
    # No new checkpoint written -- still exactly the one that was already there.
    checkpoints = list((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) == 1


def test_run_training_ignores_checkpoint_with_mismatched_config_hash(tmp_path: Path) -> None:
    from torch.optim import AdamW

    config_a = _tiny_config(max_steps=2, checkpoint_interval_steps=2)
    config_b = _tiny_config(max_steps=2, checkpoint_interval_steps=2)
    config_b.seed = 1  # top-level field -- different config_hash than config_a
    dataloader = torch.utils.data.DataLoader(_TinySegDataset(), batch_size=2, shuffle=True)

    supernet = PaceSegSupernet(config_a.supernet)
    optimizer = AdamW(supernet.parameters(), lr=config_a.training.lr)
    save_checkpoint(tmp_path / "checkpoints" / "step_00000002.pt", supernet, optimizer, step=2, config=config_a)

    buffer = io.StringIO()
    console = Console(file=buffer, width=200)
    run_training(config_b, dataloader, output_dir=tmp_path, console=console, device="cpu")

    assert "ignoring it" in buffer.getvalue()
    # Started from scratch and ran the full budget again -- a second, later checkpoint
    # is written on top of the mismatched one, not silently skipped.
    checkpoints = sorted((tmp_path / "checkpoints").glob("*.pt"))
    assert len(checkpoints) >= 1
    assert load_checkpoint(checkpoints[-1])["config_hash"] == config_b.config_hash()
