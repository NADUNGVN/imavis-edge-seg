"""Pre-registered grids and selection rules for the router progressive ablation
(`docs/COORDINATION_LOG.md` open thread #2), locked with Codex 2026-09-21. Every
value here must be computed from fit-half data only, before the held-out half is
read, or the whole "pre-registered" discipline this project already applies to the
QAT screen's decision thresholds is void here too.
"""

from __future__ import annotations

import numpy as np

RISK_TARGET_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def macro_quantile_grid(
    per_split_pooled_values: list[np.ndarray | list[float]],
    quantiles: tuple[float, ...] = RISK_TARGET_QUANTILES,
) -> list[float]:
    """One grid value per `quantiles` entry, macro-averaged across splits: for each
    split, compute the quantile over that split's own pooled fit-half values (e.g.
    all 4 candidate levels' per-image observed error, or the probe's raw risk score,
    concatenated), then average the quantile *value* across splits with equal weight
    per split -- the same macro-averaging convention this project already uses for
    "mean across N splits" headline numbers (README/reports), so a split with more
    images doesn't dominate one with fewer. Deduplicated and sorted ascending: two
    quantiles that happen to collapse to the same value produce one grid point, not
    a spurious extra operating point."""
    if not per_split_pooled_values:
        raise ValueError("need at least one split's pooled values")
    per_split_quantiles = np.stack(
        [np.quantile(np.asarray(values, dtype=np.float64), quantiles) for values in per_split_pooled_values]
    )  # (num_splits, num_quantiles)
    macro = per_split_quantiles.mean(axis=0)
    return sorted({float(v) for v in macro})


def select_budget_matched_operating_point(
    operating_points: list[tuple[float, float, object]], budget_ms: float
) -> tuple[object, bool]:
    """Pre-registered selection rule (Codex, 2026-09-21): `operating_points` is a
    list of `(fit_half_mean_latency_ms, fit_half_quality, label)` triples -- both
    numbers measured on the fit-half only (never held-out). Among the points whose
    fit-half mean latency does not exceed `budget_ms`, return the one with the
    highest fit-half quality (ties broken by lower latency), with `feasible=True`.
    If none is feasible, return the lowest-latency point with `feasible=False` -- the
    caller still evaluates it on held-out and reports the resulting violation rate as
    a sanity check, it just isn't a genuine budget-respecting choice."""
    if not operating_points:
        raise ValueError("need at least one operating point to choose from")
    feasible = [p for p in operating_points if p[0] <= budget_ms]
    if feasible:
        best = max(feasible, key=lambda p: (p[1], -p[0]))
        return best[2], True
    cheapest = min(operating_points, key=lambda p: p[0])
    return cheapest[2], False
