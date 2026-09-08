from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from imavis_edge_seg.config import ExperimentConfig  # noqa: E402
from imavis_edge_seg.models import PaceSegSupernet, extract_subnet  # noqa: E402

# 16 divides the four stride-2 reductions (stem + 3 stages) exactly; every configured
# input_resolution in config.py is chosen to satisfy this too.
_TEST_HW = (64, 64)


@pytest.fixture
def supernet() -> PaceSegSupernet:
    config = ExperimentConfig(experiment_id="test")
    return PaceSegSupernet(config.supernet)


@pytest.mark.parametrize("level", ["tiny", "small", "medium", "large"])
def test_supernet_forward_shape(supernet: PaceSegSupernet, level: str) -> None:
    x = torch.randn(2, 3, *_TEST_HW)
    out = supernet(x, level)  # type: ignore[arg-type]
    assert out.shape == (2, supernet.config.num_classes, *_TEST_HW)


def test_larger_levels_have_more_parameters(supernet: PaceSegSupernet) -> None:
    def count_active_params(level: str) -> int:
        subnet = extract_subnet(supernet, level)  # type: ignore[arg-type]
        return sum(p.numel() for p in subnet.parameters())

    counts = {level: count_active_params(level) for level in ["tiny", "small", "medium", "large"]}
    assert counts["tiny"] < counts["small"] < counts["medium"] < counts["large"]


@pytest.mark.parametrize("level", ["tiny", "small", "medium", "large"])
def test_extracted_subnet_matches_supernet_output(supernet: PaceSegSupernet, level: str) -> None:
    supernet.eval()
    x = torch.randn(1, 3, *_TEST_HW)
    with torch.no_grad():
        supernet_out = supernet(x, level)  # type: ignore[arg-type]
        subnet = extract_subnet(supernet, level)  # type: ignore[arg-type]
        subnet_out = subnet(x)
    assert subnet_out.shape == supernet_out.shape
    assert torch.allclose(subnet_out, supernet_out, atol=1e-5)


def test_extracted_subnet_exports_to_onnx(supernet: PaceSegSupernet, tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from imavis_edge_seg.export import export_subnet_onnx

    subnet = extract_subnet(supernet, "tiny")
    out_path = export_subnet_onnx(subnet, tmp_path / "tiny.onnx", *_TEST_HW)  # type: ignore[arg-type]
    assert out_path.exists()
    onnx.checker.check_model(str(out_path))
