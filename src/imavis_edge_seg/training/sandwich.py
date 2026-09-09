"""Sandwich rule level sampling (RESEARCH_PLAN.md §5.3 A): every training step always
runs the largest ("teacher", the in-place distillation source) and the smallest level,
plus a few random middle levels, so the shared weights get gradient signal from the
whole elasticity range without paying for every level every step.
"""

from __future__ import annotations

import random

from imavis_edge_seg.config import ElasticityLevel, SupernetConfig, TrainingConfig


def sample_training_levels(
    supernet_config: SupernetConfig, training_config: TrainingConfig, rng: random.Random
) -> list[ElasticityLevel]:
    """Returns levels in teacher-first order (largest first) so the caller can use the
    first entry's logits as the in-place distillation target for the rest."""
    levels = supernet_config.levels
    if len(levels) == 1:
        return list(levels)

    largest = levels[-1]
    smallest = levels[0]
    middle = levels[1:-1]
    n = min(training_config.sandwich_num_random_middle, len(middle))
    sampled_middle = rng.sample(middle, n) if n > 0 else []

    ordered = [largest, *sampled_middle, smallest]
    seen: set[ElasticityLevel] = set()
    result: list[ElasticityLevel] = []
    for level in ordered:
        if level not in seen:
            seen.add(level)
            result.append(level)
    return result
