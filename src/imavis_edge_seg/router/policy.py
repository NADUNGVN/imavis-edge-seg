"""Router decision policy (`docs/RESEARCH_PLAN.md` Contribution 3 / RQ3): given a
risk estimate and a `RouterConfig`, picks which elasticity level to run. Reuses
`search.pareto.ParetoPoint` for the available (level, cost, quality) candidates on one
deployment target -- the router's job is choosing among already-known candidates, not
recomputing latency/mIoU itself; `candidates` must already be scoped to one
`(device_id, backend)`, the same convention `search.pareto` uses.

`docs/COORDINATION_LOG.md` open thread #2 tracks a progressive ablation, not a clean
2x2 (cell D needs an extra axis -- an explicit hardware latency budget -- the other
cells don't have):
- **A** (`calibrated_risk`/`entropy`): one risk estimate per image/window (from a
  single cheap probe pass), escalation is a rank-step heuristic (`_select_by_risk`).
- **B** (`latency_spacing_risk`): same single risk estimate, but escalation targets a
  latency *magnitude* instead of a rank step (`_select_by_latency_spacing`) -- still a
  heuristic ablation on using latency magnitude, not a real latency optimizer.
- **C** (`candidate_specific_risk`): a *separate* predicted error per candidate level,
  each from an independent calibrator fit on the same shared probe signal (see
  `calibrator.fit_per_level_calibrators`) -- `_select_by_candidate_specific_risk`.
- **D** (`risk_latency_constrained`): the actual full method -- per-candidate
  predicted error *and* an explicit per-device latency budget
  (`RouterConfig.latency_budget_ms`) -- `_select_by_risk_and_latency_budget`.
"""

from __future__ import annotations

from imavis_edge_seg.config import ElasticityLevel, RouterConfig
from imavis_edge_seg.search.pareto import ParetoPoint


def select_level(
    candidates: list[ParetoPoint],
    config: RouterConfig,
    calibrated_risk: float | None = None,
    raw_risk: float | None = None,
    per_level_risk: dict[ElasticityLevel, float] | None = None,
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
    if config.strategy == "latency_spacing_risk":
        if calibrated_risk is None:
            raise ValueError("strategy='latency_spacing_risk' requires calibrated_risk")
        return _select_by_latency_spacing(ordered, calibrated_risk, config.risk_target)
    if config.strategy == "candidate_specific_risk":
        if per_level_risk is None:
            raise ValueError("strategy='candidate_specific_risk' requires per_level_risk")
        return _select_by_candidate_specific_risk(ordered, per_level_risk, config.risk_target)
    if config.strategy == "risk_latency_constrained":
        if per_level_risk is None:
            raise ValueError("strategy='risk_latency_constrained' requires per_level_risk")
        if config.latency_budget_ms is None:
            raise ValueError("strategy='risk_latency_constrained' requires config.latency_budget_ms")
        return _select_by_risk_and_latency_budget(
            ordered, per_level_risk, config.risk_target, config.latency_budget_ms
        )
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


def _select_by_latency_spacing(ordered: list[ParetoPoint], risk: float, risk_target: float) -> ElasticityLevel:
    """Cell B ("latency-spacing-aware escalation", locked with Codex 2026-09-21 --
    deliberately not "latency-value cost": still a heuristic ablation, not a real
    latency optimizer). `r = max(1, risk / risk_target)`, `t_target = clip(t_min * r,
    t_min, t_max)`, then pick the candidate whose latency is *closest* to `t_target`
    (ties broken toward the cheaper candidate) -- closest-match instead of
    "cheapest >= target" so a candidate with an unusually large latency jump doesn't
    get selected just because it's the first to clear a threshold. When `risk <=
    risk_target`, `r == 1` and `t_target == t_min`, so this already reduces to picking
    the cheapest candidate without a separate branch."""
    if risk_target <= 0:
        raise ValueError(f"risk_target must be positive, got {risk_target!r}")
    t_min = ordered[0].latency_ms
    t_max = ordered[-1].latency_ms
    r = max(1.0, risk / risk_target)
    t_target = min(max(t_min * r, t_min), t_max)
    return min(ordered, key=lambda p: (abs(p.latency_ms - t_target), p.latency_ms)).level


def _select_by_candidate_specific_risk(
    ordered: list[ParetoPoint], per_level_risk: dict[ElasticityLevel, float], risk_target: float
) -> ElasticityLevel:
    """Cell C: `per_level_risk[level]` is that level's *own* predicted error (from its
    own calibrator, `calibrator.fit_per_level_calibrators`), all derived from the same
    shared probe signal. Rank-only cost: scan candidates cheapest-first, return the
    first whose own predicted error is at or below `risk_target`. If none qualify,
    fall back to the most expensive (safest) candidate -- mirrors `_select_by_risk`'s
    escalate-to-largest-on-no-match behavior; not explicitly specified by Codex for
    this cell, flagged in docs/COORDINATION_LOG.md for confirmation."""
    for point in ordered:
        if per_level_risk[point.level] <= risk_target:
            return point.level
    return ordered[-1].level


def _select_by_risk_and_latency_budget(
    ordered: list[ParetoPoint],
    per_level_risk: dict[ElasticityLevel, float],
    risk_target: float,
    latency_budget_ms: float,
) -> ElasticityLevel:
    """Cell D ("risk-and-latency-constrained policy", the actual full method -- Codex
    rewrote this cell 2026-09-21 after flagging that a naive "candidate-specific risk
    filtered, then cheapest" design was operationally identical to cell C whenever any
    candidate met the risk target). `S_B` = candidates within `latency_budget_ms`:
    (1) if any candidate in `S_B` meets `risk_target`, pick the cheapest of those;
    (2) else pick the in-budget candidate with the lowest predicted error;
    (3) ties broken by lower latency;
    (4) if `S_B` is empty, pick the cheapest candidate overall.
    Deliberately no quality-gain-per-ms fallback (rejected by Codex: depends on the
    starting point, can pick a Pareto-dominated candidate, hard to justify to a
    reviewer)."""
    in_budget = [p for p in ordered if p.latency_ms <= latency_budget_ms]
    if not in_budget:
        return ordered[0].level
    meeting_target = [p for p in in_budget if per_level_risk[p.level] <= risk_target]
    if meeting_target:
        return min(meeting_target, key=lambda p: p.latency_ms).level
    return min(in_budget, key=lambda p: (per_level_risk[p.level], p.latency_ms)).level
