"""Nested bootstrap for the D-vs-A and D-vs-A-hard deltas (2026-10-03).

Each replicate resamples the fit half AND the held-out half of every split (with
replacement), refits the per-candidate calibrators and the macro risk-target grid
on the resampled fit halves, re-selects every operating point on the resampled fit
halves, and evaluates on the resampled held-out halves. The resulting interval
therefore includes fit-half calibration and operating-point-selection uncertainty,
unlike the held-out-only bootstrap. Median route-cost tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from router_review_analyses import BASE, LEVELS, RUNS, Split, load_costs, run_policy

from imavis_edge_seg.router.calibrator import fit_risk_calibrator
from imavis_edge_seg.router.grid import macro_quantile_grid


def resample(sp: Split, fi: np.ndarray, ti: np.ndarray) -> Split:
    new = Split.__new__(Split)
    new.probe = sp.probe
    new.fit_scores = sp.fit_scores[fi]
    new.test_scores = sp.test_scores[ti]
    new.fit_cm = {lv: sp.fit_cm[lv][fi] for lv in LEVELS}
    new.test_cm = {lv: sp.test_cm[lv][ti] for lv in LEVELS}
    new.fit_err = {lv: sp.fit_err[lv][fi] for lv in LEVELS}
    new.test_err = {lv: sp.test_err[lv][ti] for lv in LEVELS}
    new.cal = {lv: fit_risk_calibrator(new.fit_scores, new.fit_err[lv]) for lv in LEVELS}
    new.pooled = new.cal
    return new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    costs = load_costs("median")
    splits = {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
    names = list(splits["a"])
    rng = np.random.default_rng(args.seed)
    d_a, d_ah = [], []
    for rep in range(args.reps):
        idx = {n: (rng.integers(0, len(splits["a"][n].fit_scores), len(splits["a"][n].fit_scores)),
                   rng.integers(0, len(splits["a"][n].test_scores), len(splits["a"][n].test_scores))) for n in names}
        da, dah = [], []
        for run in RUNS:
            rs = {n: resample(splits[run][n], *idx[n]) for n in names}
            grid = macro_quantile_grid([np.concatenate([rs[n].fit_err[lv] for lv in LEVELS]) for n in names])
            for cost in costs.values():
                budgets = sorted(set(cost.values()))
                for n in names:
                    sp = rs[n]
                    D = run_policy("D", sp, grid, budgets, cost, sp.cal)
                    A = run_policy("A", sp, grid, budgets, cost, sp.cal)
                    AH = run_policy("A_hard", sp, grid, budgets, cost, sp.cal)
                    for b in budgets:
                        dah.append(D[b]["miou"] - AH[b]["miou"])
                        if A[b]["violation"] == 0.0:
                            da.append(D[b]["miou"] - A[b]["miou"])
        d_a.append(np.mean(da) * 100)
        d_ah.append(np.mean(dah) * 100)
        if (rep + 1) % 10 == 0:
            print(f"rep {rep + 1}: D-A {np.mean(d_a):.3f}  D-Ahard {np.mean(d_ah):.3f}", flush=True)
    out = {
        "reps": args.reps,
        "D_minus_A_fair_points": {"mean": float(np.mean(d_a)), "ci95": [float(x) for x in np.percentile(d_a, [2.5, 97.5])]},
        "D_minus_A_hard_points": {"mean": float(np.mean(d_ah)), "ci95": [float(x) for x in np.percentile(d_ah, [2.5, 97.5])]},
        "note": "fit and held-out halves resampled; calibrators, risk grid, and operating points refit per replicate",
    }
    Path("reports/router_nested_bootstrap_20261003.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
