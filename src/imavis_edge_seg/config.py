"""Experiment configuration schema.

Kept intentionally close to the four contributions in `docs/RESEARCH_PLAN.md`: an
elastic supernet search space, a hardware-in-the-loop search objective, a calibrated
router, and dataset/backend targets for the cross-platform benchmark protocol.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

Backend = Literal["tensorrt_gpu", "xavier_dla", "hailo_hef", "onnxruntime_cpu"]
ElasticityLevel = Literal["tiny", "small", "medium", "large"]
AdverseCondition = Literal["clean", "night", "rain", "fog", "snow"]


def _default_levels() -> list[ElasticityLevel]:
    return ["tiny", "small", "medium", "large"]


def _default_width_multipliers() -> dict[ElasticityLevel, float]:
    return {"tiny": 0.25, "small": 0.5, "medium": 0.75, "large": 1.0}


def _default_depth_blocks() -> dict[ElasticityLevel, int]:
    return {"tiny": 2, "small": 3, "medium": 4, "large": 6}


def _default_input_resolutions() -> dict[ElasticityLevel, tuple[int, int]]:
    return {
        "tiny": (384, 192),
        "small": (512, 256),
        "medium": (768, 384),
        "large": (1024, 512),
    }


def _default_target_backends() -> list[Backend]:
    # xavier_dla deliberately excluded from the default Pareto-search/headline-claim
    # backend set: confirmed live 2026-09-08/10 that Xavier's DLA has a hard 16-subgraph-
    # per-core budget the encoder alone already exhausts, so the decoder never gets a
    # DLA slot (see reports/edge/E3_compiler_smoke_test_20260908.md). This is
    # RESEARCH_PLAN.md §11's own "excessive DLA fallback -> demote to secondary
    # ablation, keep TensorRT GPU + Hailo as the two primary backends" contingency,
    # executed 2026-09-10 -- DLA is still compiled/tested (encoder-only) as a secondary
    # ablation, just not part of the default search/comparison target set.
    return ["tensorrt_gpu", "hailo_hef"]


class SupernetConfig(BaseModel):
    """Elastic search space. Restricted to the compiler-safe operator intersection --
    do not add operators here without a smoke-test result recorded per backend."""

    levels: list[ElasticityLevel] = Field(default_factory=_default_levels)
    width_multipliers: dict[ElasticityLevel, float] = Field(default_factory=_default_width_multipliers)
    depth_blocks: dict[ElasticityLevel, int] = Field(default_factory=_default_depth_blocks)
    input_resolutions: dict[ElasticityLevel, tuple[int, int]] = Field(
        default_factory=_default_input_resolutions
    )
    num_classes: int = 19  # Cityscapes trainId classes
    compiler_safe_ops_only: bool = True


class SearchConfig(BaseModel):
    """Hardware-in-the-loop Pareto search objective weights (RQ1)."""

    # "measured_latency", not "..._energy": no external calibrated power meter is
    # available yet (deferred 2026-09-11, see RESEARCH_PLAN.md §14 "Power meter /
    # camera domain") -- energy/J-frame must never be claimed from telemetry alone.
    # §11's Go bar ("cuts p95 latency OR J/frame by >=20%") still holds latency-only.
    objective: Literal["flops", "measured_latency", "measured_latency_energy"] = "measured_latency"
    lambda_latency: float = 1.0
    mu_energy: float = 1.0
    alpha_distill: float = 0.5
    beta_calibration: float = 0.25
    target_backends: list[Backend] = Field(default_factory=_default_target_backends)


class RouterConfig(BaseModel):
    """Calibrated visual-risk router (RQ3). Thresholds must be fit on validation splits."""

    enabled: bool = True
    strategy: Literal[
        "static_small",
        "static_large",
        "oracle",
        "entropy",
        "calibrated_risk",
        "latency_spacing_risk",
        "candidate_specific_risk",
        "risk_latency_constrained",
    ] = "calibrated_risk"
    window_frames: int = 16  # amortize engine-switch overhead over this many frames
    risk_target: float = 0.05
    # Only used by strategy="risk_latency_constrained" (cell D, docs/COORDINATION_LOG.md
    # open thread #2): an explicit per-device hardware latency budget, evaluated
    # alongside risk_target as a pre-registered (budget, risk_target) grid.
    latency_budget_ms: float | None = None


def _default_calibration_conditions() -> list[AdverseCondition]:
    return ["clean", "night", "rain", "fog", "snow"]


class QuantizationConfig(BaseModel):
    precision: Literal["fp32", "fp16", "int8_ptq", "int8_qat"] = "int8_qat"
    calibration_conditions: list[AdverseCondition] = Field(default_factory=_default_calibration_conditions)


class TrainingConfig(BaseModel):
    """Sandwich-rule supernet training (RESEARCH_PLAN.md §5.3 A). Distillation weight
    reuses `SearchConfig.alpha_distill` rather than duplicating it here."""

    batch_size: int = 8
    num_workers: int = 4
    max_steps: int = 100_000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_schedule: Literal["constant", "cosine", "poly"] = "cosine"
    warmup_steps: int = 500
    sandwich_num_random_middle: int = 1  # besides the always-sampled smallest+largest
    boundary_loss_weight: float = 1.0
    distillation_temperature: float = 1.0
    grad_clip_norm: float = 5.0
    log_interval_steps: int = 50
    checkpoint_interval_steps: int = 1000
    augment: bool = True  # data.transforms.SegmentationTrainAugment vs. plain resize


class DatasetConfig(BaseModel):
    name: Literal["cityscapes", "acdc", "dark_zurich", "bdd100k"]
    root: Path
    split: str = "train"


class DeploymentTargetConfig(BaseModel):
    device_id: str  # matches ../../docs/SHARED_INFRASTRUCTURE.md §3.1 device IDs (E1, E2, ...)
    backend: Backend
    power_mode_watts: float | None = None


class ExperimentConfig(BaseModel):
    experiment_id: str
    seed: int = 0
    supernet: SupernetConfig = Field(default_factory=SupernetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    datasets: list[DatasetConfig] = Field(default_factory=list)
    deployment_targets: list[DeploymentTargetConfig] = Field(default_factory=list)
    output_root: Path = Path("outputs")

    def config_hash(self) -> str:
        payload = self.model_dump_json(exclude={"experiment_id"}).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = OmegaConf.load(path)
    if overrides:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(overrides))
    resolved = OmegaConf.to_container(raw, resolve=True)
    return ExperimentConfig.model_validate(resolved)
