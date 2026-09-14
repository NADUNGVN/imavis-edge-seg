"""Router decision policy (`docs/RESEARCH_PLAN.md` Contribution 3 / RQ3): given a
risk estimate and a `RouterConfig`, picks which elasticity level to run. Reuses
`search.pareto.ParetoPoint` for the available (level, cost, quality) candidates on one
deployment target -- the router's job is choosing among already-known candidates, not
recomputing latency/mIoU itself; `candidates` must already be scoped to one
`(device_id, backend)`, the same convention `search.pareto` uses.

MVP limitation, not hidden: this first pass has only *one* risk estimate per image/
window (from a single cheap pass, typically at the smallest level), not a per-level
risk prediction -- so "how far to escalate" above `risk_target` is a simple
proportional heuristic (`_select_by_risk`), not a calibrated per-level decision.
Refining this needs either a risk estimate conditioned on each candidate level, or
calibration data relating the cheap-level risk score to each *other* level's expected
error, neither of which exists yet.
"""

from __future__ import annotations

from imavis_edge_seg.config import ElasticityLevel, RouterConfig
from imavis_edge_seg.search.pareto import ParetoPoint


def select_level(
    candidates: list[ParetoPoint],
    config: RouterConfig,
    calibrated_risk: float | None = None,
    raw_risk: float | None = None,
) -> ElasticityLevel:
    if not candidates:
        raise ValueError("no candidate levels to route between")
    ordered = sorted(candidates, key=lambda p: p.latency_ms)

    if config.strategy == "static_small":
        return ordered[0].level
    if config.strategy == "static_large":
        return ordered[-1].level
    if config.strategy == "oracle":
        # Best mIoU regardless of cost -- a research upper bound for comparison
        # (RESEARCH_PLAN.md §7's baseline axes), not a deployable real-time policy.
        return max(candidates, key=lambda p: p.miou).level
    if config.strategy == "entropy":
        if raw_risk is None:
            raise ValueError("strategy='entropy' requires raw_risk")
        return _select_by_risk(ordered, raw_risk, config.risk_target)
    if config.strategy == "calibrated_risk":
        if calibrated_risk is None:
            raise ValueError("strategy='calibrated_risk' requires calibrated_risk")
        return _select_by_risk(ordered, calibrated_risk, config.risk_target)
    raise ValueError(f"unknown router strategy {config.strategy!r}")


def _select_by_risk(ordered: list[ParetoPoint], risk: float, risk_target: float) -> ElasticityLevel:
    """Risk at or below target: use the cheapest (smallest) level. Above target:
    escalate to a larger level, one step per whole multiple of `risk_target` the
    observed risk represents, capped at the largest available level."""
    if risk <= risk_target:
        return ordered[0].level
    overshoot_ratio = risk / risk_target
    index = min(int(overshoot_ratio), len(ordered) - 1)
    return ordered[index].level
