from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from imavis_edge_seg.config import ExperimentConfig  # noqa: E402
from imavis_edge_seg.models.baselines import BASELINE_NAMES, build_baseline_model  # noqa: E402
from imavis_edge_seg.training.baseline_trainer import run_baseline_training  # noqa: E402
from imavis_edge_seg.training.checkpoint import load_checkpoint  # noqa: E402

_TEST_HW = (64, 64)


def test_build_baseline_model_mobilenetv3_deeplabv3_output_shape() -> None:
    model = build_baseline_model("mobilenetv3_deeplabv3", num_classes=19)
    image = torch.randn(2, 3, *_TEST_HW)
    logits = model(image)
    assert logits.shape == (2, 19, *_TEST_HW)


@pytest.mark.parametrize("name", [n for n in BASELINE_NAMES if n != "mobilenetv3_deeplabv3"])
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
