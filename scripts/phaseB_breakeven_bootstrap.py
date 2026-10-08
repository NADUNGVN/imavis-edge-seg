"""Phase B: image-level bootstrap interval for the counterfactual break-even overhead.

Re-implements the mean-cost part of router_breakeven.py (mixture_gain over the same
alpha grid, same 25-target dense grid, same calibrators) but evaluates it on bootstrap
resamples of the held-out images (per split, shared across runs and devices, seed 0,
REPS replicates). Choices depend only on probe scores and calibrators, never on labels,
so they are computed once; each replicate re-sums the resampled confusion matrices and
latencies. The break-even overhead is where the mean gain crosses zero, linearly
interpolated between alpha grid points, in ms of mean overhead (as in the main analysis).
Point estimate = full-sample value; interval = percentile 2.5/97.5. Image sampling only.
Output: reports/phaseB_breakeven_bootstrap_20261004.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from router_mean_budget_frontier import c_choice
from router_review_analyses import BASE, LEVELS, RUNS, Split, miou, ordered_levels
from router_same_harness_analysis import tables_from
from router_breakeven import CONFIGS, ALPHAS

REPS = 200


def precompute(splits, rc_order):
    items = []
    for run in RUNS:
        for s, sp in splits[run].items():
            preds = np.concatenate([[sp.cal[lv].predict(x) for x in sp.fit_scores] for lv in LEVELS])
            targets = sorted(set(np.quantile(preds, np.linspace(0.02, 0.98, 25)).tolist()))
            risk = [{lv: sp.cal[lv].predict(x) for lv in LEVELS} for x in sp.test_scores]
            for t in targets:
                ch = np.array([LEVELS.index(c_choice(r, t, rc_order)) for r in risk])
                items.append((s, sp, ch))
    return items


def gain_curve(items, sc, over, idx):
    so = ordered_levels(sc)
    out = []
    for a in ALPHAS:
        rc = {lv: sc[lv] + a * over[lv] for lv in LEVELS}
        cost = np.array([rc[lv] for lv in LEVELS])
        gains = []
        for s, sp, ch in items:
            ii = idx[s]
            stack = np.stack([sp.test_cm[lv][ii] for lv in LEVELS])  # L, n, C, C
            c = ch[ii]
            cms = stack[c, np.arange(len(ii))].sum(0)
            m = float(cost[c].mean())
            if m < sc[so[0]]:
                continue
            if m > sc[so[-1]]:
                ref = miou(sp.test_cm[so[-1]][ii].sum(0))
            else:
                for lo, hi in zip(so, so[1:]):
                    if sc[lo] <= m <= sc[hi]:
                        p = (m - sc[lo]) / (sc[hi] - sc[lo])
                        ref = miou((1 - p) * sp.test_cm[lo][ii].sum(0) + p * sp.test_cm[hi][ii].sum(0))
                        break
            gains.append((miou(cms) - ref) * 100)
        out.append(float(np.mean(gains)))
    return out


def crossing(ovs, gains):
    for k in range(1, len(gains)):
        if gains[k - 1] >= 0 > gains[k]:
            t = gains[k - 1] / (gains[k - 1] - gains[k])
            return ovs[k - 1] + t * (ovs[k] - ovs[k - 1])
    return None if gains[0] < 0 else float("inf")


def main() -> None:
    splits = {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
    sizes = {s: len(sp.test_scores) for s, sp in splits["a"].items()}
    rng = np.random.default_rng(0)
    boots = [{s: rng.integers(0, n, n) for s, n in sizes.items()} for _ in range(REPS)]
    full = {s: np.arange(n) for s, n in sizes.items()}
    out = {"_definition": __doc__, "reps": REPS}
    for dev, (path, suffix) in CONFIGS.items():
        rc_full, sc = tables_from(Path(path), "median", suffix)
        over = {lv: rc_full[lv] - sc[lv] for lv in LEVELS}
        ovs = [float(np.mean(list(over.values())) * a) for a in ALPHAS]
        items = precompute(splits, ordered_levels(rc_full))
        point_curve = gain_curve(items, sc, over, full)
        be = crossing(ovs, point_curve)
        bes, zero_gain = [], []
        for k, idx in enumerate(boots):
            g = gain_curve(items, sc, over, idx)
            bes.append(crossing(ovs, g))
            zero_gain.append(g[0])
            if k % 20 == 0:
                print(dev, k, flush=True)
        finite = [b for b in bes if b is not None and np.isfinite(b)]
        out[dev] = {"alphas": ALPHAS, "mean_overhead_ms": ovs, "point_gain_curve": point_curve,
                    "breakeven_ms": be, "breakeven_ci95_ms": [float(x) for x in np.percentile(finite, [2.5, 97.5])],
                    "replicates_with_crossing": len(finite), "replicates_no_gain_even_at_zero": sum(b is None for b in bes),
                    "zero_overhead_gain_ci95": [float(x) for x in np.percentile(zero_gain, [2.5, 97.5])]}
        print(dev, out[dev]["breakeven_ms"], out[dev]["breakeven_ci95_ms"], flush=True)
    Path("reports/phaseB_breakeven_bootstrap_20261004.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
