"""Selective-prediction-style metrics for the router progressive ablation
(`docs/COORDINATION_LOG.md` open thread #2): assess a risk *signal*'s quality
(does higher predicted risk actually track higher realized error?) and a
routing *policy*'s cost behavior (how often does it exceed a device's latency
budget?), independent of `router.policy`'s decision logic itself.

`area_under_risk_coverage`/`risk_at_coverage` are the standard selective-
classification formulation (Geifman & El-Yaniv): rank samples by risk score
ascending (most-confident first), then track the average realized error among
the lowest-risk `coverage` fraction as `coverage` grows from 0 to 1. A good
risk signal keeps that average low at low coverage (its most-confident
predictions really are its most correct ones) -- this is orthogonal to which
elasticity level a policy picks, deliberately: it can score any (risk, error)
pairing, e.g. the probe's own risk vs. its own error (signal quality) or a
per-candidate predicted risk vs. the policy's actually-realized error (decision
quality), so it is not itself Pareto-aware and is reported alongside, not
instead of, the quality-latency frontier and budget-violation rate.
"""

from __future__ import annotations

import numpy as np


def _sorted_errors(risk_scores: np.ndarray | list[float], errors: np.ndarray | list[float]) -> np.ndarray:
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if len(risk_scores) == 0:
        raise ValueError("cannot compute a risk-coverage metric with zero samples")
    if len(risk_scores) != len(errors):
        raise ValueError(f"risk_scores ({len(risk_scores)}) and errors ({len(errors)}) length mismatch")
    order = np.argsort(risk_scores, kind="stable")
    return errors[order]


def risk_at_coverage(
    risk_scores: np.ndarray | list[float], errors: np.ndarray | list[float], coverage: float
) -> float:
    """Average realized error among the `coverage` fraction of samples with the
    *lowest* risk score (e.g. `coverage=0.5` -> the risk signal's most-confident
    half). `coverage` must be in `(0, 1]`."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")
    sorted_errors = _sorted_errors(risk_scores, errors)
    k = max(1, round(coverage * len(sorted_errors)))
    return float(sorted_errors[:k].mean())


def area_under_risk_coverage(risk_scores: np.ndarray | list[float], errors: np.ndarray | list[float]) -> float:
    """AURC: the average of `risk_at_coverage` over every achievable coverage level
    `k/N` for `k = 1..N` -- lower is better (a perfect risk signal front-loads all
    the low-error samples, keeping the running average near its floor for as long
    as possible)."""
    sorted_errors = _sorted_errors(risk_scores, errors)
    cumulative_mean = np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)
    return float(cumulative_mean.mean())


def budget_violation_rate(chosen_latencies_ms: np.ndarray | list[float], budget_ms: float) -> float:
    """Fraction of routed images whose chosen candidate's latency exceeded
    `budget_ms` -- a hard constraint violation, distinct from an accuracy/quality
    tradeoff. Strict `>`: a chosen latency exactly at the budget is not a
    violation."""
    chosen_latencies_ms = np.asarray(chosen_latencies_ms, dtype=np.float64)
    if len(chosen_latencies_ms) == 0:
        raise ValueError("cannot compute a violation rate with zero samples")
    return float((chosen_latencies_ms > budget_ms).mean())
