from pathlib import Path

from imavis_edge_seg.config import ExperimentConfig, load_config


def test_default_config_round_trips() -> None:
    cfg = ExperimentConfig(experiment_id="unit-test")
    assert cfg.supernet.levels == ["tiny", "small", "medium", "large"]
    assert cfg.search.objective == "measured_latency_energy"
    assert len(cfg.config_hash()) == 12


def test_load_default_yaml() -> None:
    cfg = load_config(Path(__file__).parents[1] / "configs" / "experiment" / "default.yaml")
    assert cfg.experiment_id == "pace_seg_dev_smoke"
    assert cfg.router.strategy == "calibrated_risk"
    assert {d.name for d in cfg.datasets} == {"cityscapes", "acdc"}


def test_config_override() -> None:
    cfg = load_config(
        Path(__file__).parents[1] / "configs" / "experiment" / "default.yaml",
        overrides=["seed=7"],
    )
    assert cfg.seed == 7
