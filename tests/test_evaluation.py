import pytest

torch = pytest.importorskip("torch")

from imavis_edge_seg.config import ExperimentConfig  # noqa: E402
from imavis_edge_seg.data.labels import IGNORE_INDEX  # noqa: E402
from imavis_edge_seg.evaluation.evaluator import evaluate_level  # noqa: E402
from imavis_edge_seg.evaluation.metrics import (  # noqa: E402
    ConfusionMatrixAccumulator,
    compute_confusion_matrix,
    iou_per_class,
)
from imavis_edge_seg.models.supernet import PaceSegSupernet  # noqa: E402

_TEST_HW = (64, 64)


def test_confusion_matrix_matches_hand_computed_example() -> None:
    # 2-class toy example: pred=[0,0,1,1], target=[0,1,1,1]
    pred = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0, 1, 1, 1])
    matrix = compute_confusion_matrix(pred, target, num_classes=2)
    # rows=target, cols=pred
    expected = torch.tensor([[1, 0], [1, 2]])
    assert torch.equal(matrix, expected)


def test_confusion_matrix_excludes_ignore_index() -> None:
    pred = torch.tensor([0, 1, 0])
    target = torch.tensor([0, IGNORE_INDEX, 0])
    matrix = compute_confusion_matrix(pred, target, num_classes=2)
    assert matrix.sum() == 2  # the ignored pixel is excluded entirely


def test_iou_per_class_perfect_prediction_is_one() -> None:
    target = torch.tensor([0, 0, 1, 1, 2])
    matrix = compute_confusion_matrix(target, target, num_classes=3)
    iou = iou_per_class(matrix)
    assert torch.allclose(iou, torch.ones(3))


def test_iou_per_class_absent_class_is_nan() -> None:
    # class 2 never appears in pred or target
    pred = torch.tensor([0, 1, 0, 1])
    target = torch.tensor([0, 1, 1, 0])
    matrix = compute_confusion_matrix(pred, target, num_classes=3)
    iou = iou_per_class(matrix)
    assert torch.isnan(iou[2])
    assert not torch.isnan(iou[0])


def test_confusion_matrix_accumulator_across_batches() -> None:
    accumulator = ConfusionMatrixAccumulator(num_classes=2)
    accumulator.update(torch.tensor([0, 0]), torch.tensor([0, 0]))
    accumulator.update(torch.tensor([1, 1]), torch.tensor([1, 0]))
    result = accumulator.compute()
    assert result.num_pixels == 4
    assert 0.0 <= result.miou <= 1.0


def test_evaluate_level_end_to_end_with_synthetic_data() -> None:
    config = ExperimentConfig(experiment_id="eval-test")
    for level in config.supernet.levels:
        config.supernet.input_resolutions[level] = _TEST_HW
    supernet = PaceSegSupernet(config.supernet)

    class _TinyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.randn(3, *_TEST_HW), torch.randint(0, 19, _TEST_HW)

    loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=2)
    result = evaluate_level(supernet, "tiny", loader, device="cpu")
    assert 0.0 <= result.miou <= 1.0 or result.miou != result.miou  # allow nan if degenerate
    assert result.num_pixels == 4 * _TEST_HW[0] * _TEST_HW[1]
    assert len(result.per_class_iou) == 19
