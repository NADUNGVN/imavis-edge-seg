"""Hardware-in-the-loop Pareto subnet selection (`docs/RESEARCH_PLAN.md` Contribution 2
/ §5.4). This is *not* a continuous architecture search -- the elastic width/depth axes
were already fixed at training time (`models/supernet.py`'s 4 levels). What's left to
decide, per deployment target (a specific device + backend), is which of the 4 already-
trained, already-compiled static levels to run under a latency budget or multi-
objective tradeoff -- using measured cost (`benchmark/lookup_table.py`'s real latency
records), not FLOPs (RQ1).

A Pareto comparison only makes sense *within* one (device_id, backend) deployment
target -- comparing E1's latency to E3's doesn't inform a real decision, since you
don't choose which device to deploy on when picking a level for a device already in
the field. Group points by (device_id, backend) before computing a frontier; the
`_per_target` helpers do this automatically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from imavis_edge_seg.config import Backend, ElasticityLevel


@dataclass(frozen=True)
class ParetoPoint:
    level: ElasticityLevel
    device_id: str
    backend: Backend
    precision: str
    latency_ms: float
    miou: float
    dataset: str  # which mIoU split (e.g. "cityscapes", "acdc/fog") miou came from


def build_pareto_points(
    lookup_rows: list[dict[str, Any]],
    miou_by_level: dict[str, dict[str, float]],
    dataset: str,
    latency_field: str = "end_to_end_p95_ms",
) -> list[ParetoPoint]:
    """Joins `lookup_table.build_lookup_table()`'s rows (real measured latency) with
    an evaluation JSON's `{level: {dataset: miou}}` shape (real measured accuracy) on
    `level`. `latency_field` defaults to p95 (tail latency, `RESEARCH_PLAN.md` §9/RQ1),
    not the mean -- pass e.g. "end_to_end_mean_ms" or "kernel_only_mean_ms" if a
    different column is wanted. Rows with a missing/empty latency value or no matching
    `(level, dataset)` mIoU entry are silently skipped (a device/level not yet
    benchmarked, or a dataset not yet evaluated at that level, is not an error here).
    """
    points: list[ParetoPoint] = []
    for row in lookup_rows:
        level = row["level"]
        latency = row.get(latency_field)
        if latency in (None, ""):
            continue
        miou = miou_by_level.get(level, {}).get(dataset)
        if miou is None:
            continue
        points.append(
            ParetoPoint(
                level=level,
                device_id=row["device_id"],
                backend=row["backend"],
                precision=row["precision"],
                latency_ms=float(latency),
                miou=float(miou),
                dataset=dataset,
            )
        )
    return points


def is_dominated(candidate: ParetoPoint, others: list[ParetoPoint]) -> bool:
    """`candidate` is dominated if some other point has latency <= candidate's *and*
    mIoU >= candidate's, with at least one strict -- i.e. some other point is at least
    as good on both axes and strictly better on one."""
    for other in others:
        if other is candidate:
            continue
        not_worse = other.latency_ms <= candidate.latency_ms and other.miou >= candidate.miou
        strictly_better = other.latency_ms < candidate.latency_ms or other.miou > candidate.miou
        if not_worse and strictly_better:
            return True
    return False


def pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Non-dominated points, sorted by ascending latency. Assumes `points` is already
    scoped to one deployment target (one device_id + backend) -- see module docstring;
    use `pareto_frontiers_per_target` to group first if it isn't."""
    frontier = [p for p in points if not is_dominated(p, points)]
    return sorted(frontier, key=lambda p: p.latency_ms)


def pareto_frontiers_per_target(
    points: list[ParetoPoint],
) -> dict[tuple[str, str], list[ParetoPoint]]:
    """Groups by (device_id, backend) and computes each group's frontier separately."""
    groups: dict[tuple[str, str], list[ParetoPoint]] = defaultdict(list)
    for point in points:
        groups[(point.device_id, point.backend)].append(point)
    return {key: pareto_frontier(group) for key, group in groups.items()}


def select_under_latency_budget(
    points: list[ParetoPoint], budget_ms: float
) -> ParetoPoint | None:
    """Highest-mIoU point with latency_ms <= budget_ms (ties broken by lower latency);
    `None` if nothing fits the budget. `points` should already be scoped to one
    deployment target, same as `pareto_frontier`."""
    candidates = [p for p in points if p.latency_ms <= budget_ms]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.miou, -p.latency_ms))
