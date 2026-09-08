"""Per-level channel plan for the elastic supernet.

Channel counts are precomputed once per stage/level so every block that shares
a logical stage (stem, stage1, stage2, stage3) agrees on the same channel count
for a given elasticity level -- computing widths ad hoc inside each block would
risk two blocks disagreeing on the boundary between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from imavis_edge_seg.config import ElasticityLevel, SupernetConfig

# Base ("large", multiplier 1.0) channel counts per stage. Chosen divisible by 8
# for reasonable INT8/accelerator alignment; not tuned for accuracy yet.
BASE_STEM_CHANNELS = 32
BASE_STAGE_CHANNELS = (48, 64, 96)


def round_channels(base: int, multiplier: float, divisor: int = 8) -> int:
    """MobileNet-style channel rounding: nearest multiple of `divisor`, never
    dropping more than 10% below the unrounded target."""
    value = max(divisor, int(base * multiplier + divisor / 2) // divisor * divisor)
    if value < 0.9 * base * multiplier:
        value += divisor
    return value


@dataclass(frozen=True)
class ChannelPlan:
    stem: dict[ElasticityLevel, int] = field(default_factory=dict)
    stage1: dict[ElasticityLevel, int] = field(default_factory=dict)
    stage2: dict[ElasticityLevel, int] = field(default_factory=dict)
    stage3: dict[ElasticityLevel, int] = field(default_factory=dict)

    def max_channels(self, stage: str) -> int:
        per_level: dict[ElasticityLevel, int] = getattr(self, stage)
        return max(per_level.values())


def build_channel_plan(supernet_config: SupernetConfig) -> ChannelPlan:
    levels = supernet_config.levels
    multipliers = supernet_config.width_multipliers
    c1_base, c2_base, c3_base = BASE_STAGE_CHANNELS
    return ChannelPlan(
        stem={lvl: round_channels(BASE_STEM_CHANNELS, multipliers[lvl]) for lvl in levels},
        stage1={lvl: round_channels(c1_base, multipliers[lvl]) for lvl in levels},
        stage2={lvl: round_channels(c2_base, multipliers[lvl]) for lvl in levels},
        stage3={lvl: round_channels(c3_base, multipliers[lvl]) for lvl in levels},
    )
