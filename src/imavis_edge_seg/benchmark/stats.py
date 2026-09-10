"""Latency statistics: percentiles + bootstrap CI, per `docs/RESEARCH_PLAN.md` §9 rule 8
("Bootstrap 95% CI for latency/energy"). No hardware dependency -- pure numpy over a
list of per-inference latency measurements already captured on-device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    n: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    ci95_low_ms: float
    ci95_high_ms: float


def bootstrap_ci(
    samples: list[float], confidence: float = 0.95, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap CI of the mean. Deterministic given `seed` so a report is
    reproducible without needing to re-run the underlying benchmark."""
    if not samples:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    data = np.asarray(samples, dtype=np.float64)
    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        resample = rng.choice(data, size=len(data), replace=True)
        resampled_means[i] = resample.mean()
    alpha = (1.0 - confidence) / 2.0
    low = float(np.quantile(resampled_means, alpha))
    high = float(np.quantile(resampled_means, 1.0 - alpha))
    return (low, high)


def compute_latency_stats(samples: list[float], seed: int = 0) -> LatencyStats:
    if not samples:
        raise ValueError("cannot compute stats over an empty sample list")
    data = np.asarray(samples, dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(samples, seed=seed)
    return LatencyStats(
        n=len(samples),
        mean_ms=float(data.mean()),
        std_ms=float(data.std(ddof=1)) if len(samples) > 1 else 0.0,
        min_ms=float(data.min()),
        max_ms=float(data.max()),
        p50_ms=float(np.percentile(data, 50)),
        p90_ms=float(np.percentile(data, 90)),
        p95_ms=float(np.percentile(data, 95)),
        p99_ms=float(np.percentile(data, 99)),
        ci95_low_ms=ci_low,
        ci95_high_ms=ci_high,
    )
