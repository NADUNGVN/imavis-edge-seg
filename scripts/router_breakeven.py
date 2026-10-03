"""Break-even routing overhead (2026-10-03).

For each same-harness configuration, the measured routing overhead per route is
O(l) = C(l) - S(l) (route cost minus direct static cost of the same candidate,
same harness). We scale it, C_alpha(l) = S(l) + alpha * O(l), alpha in [0, 1], and
recompute D / T-hard versus the best feasible static candidate (hard budgets) and
candidate-specific routing versus static mixing (mean-cost budgets). The break-even
overhead is the largest alpha (reported in ms of mean overhead) at which routing
still matches or beats static. alpha = 0 means routing pays nothing beyond the
selected candidate itself (an optimistic bound: the tiny probe would be free).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from router_review_analyses import TAG, BASE, LEVELS, RUNS, Split, run_policy
from router_review_analyses_v2 import compare, mixture_gain, static_fair, threshold_policy
from router_same_harness_analysis import tables_from

CONFIGS = {
    "E3": (f"reports/static_vs_route_E3_{TAG}.json", "|logits"),
    "E1": (f"reports/static_vs_route_E1_explicit_float32_{TAG}.json", ""),
}
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def main() -> None:
    splits, grids = {}, {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        ev = json.loads((BASE / f"run_{run}_evaluation.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
        grids[run] = {k: ev[k]["risk_target_grid"] for k in splits[run]}
    out = {}
    for dev, (path, suffix) in CONFIGS.items():
        rc_full, sc = tables_from(Path(path), "median", suffix)
        over = {lv: rc_full[lv] - sc[lv] for lv in LEVELS}
        out[dev] = {"static_ms": sc, "route_ms": rc_full, "overhead_ms": over, "by_alpha": {}}
        for a in ALPHAS:
            rc = {lv: sc[lv] + a * over[lv] for lv in LEVELS}
            budgets = sorted(set(rc.values()) | set(sc.values()))
            rows = []
            for run in RUNS:
                for split_name, sp in splits[run].items():
                    t = grids[run][split_name]
                    D = run_policy("D", sp, t, budgets, rc, sp.cal)
                    TH = threshold_policy(sp, budgets, rc)
                    ST = static_fair(sp, budgets, sc)
                    for b in budgets:
                        rows.append({"split": split_name, "D": D[b], "T_hard": TH[b], "static_S": ST[b]})
            mix = mixture_gain(splits, {dev: rc}, {dev: sc})
            res = {"mean_overhead_ms": float(np.mean(list(over.values())) * a),
                   "D_vs_static": compare(rows, "D", "static_S"),
                   "T_hard_vs_static": compare(rows, "T_hard", "static_S"),
                   "mixture_gain_points": mix["mean_gain_points"], "mixture_share_above": mix["share_above"]}
            out[dev]["by_alpha"][str(a)] = res
            print(f"{dev} alpha={a:.2f} overhead~{res['mean_overhead_ms']:.2f}ms  "
                  f"hard D-static {res['D_vs_static']['mean_delta_points']:+.2f} "
                  f"({res['D_vs_static']['wins']}/{res['D_vs_static']['ties']}/{res['D_vs_static']['losses']})  "
                  f"T-static {res['T_hard_vs_static']['mean_delta_points']:+.2f}  "
                  f"mean-cost gain {res['mixture_gain_points']:+.2f} (above {res['mixture_share_above']:.0%})", flush=True)
    Path(f"reports/router_breakeven_{TAG}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote reports/router_breakeven_{TAG}.json")


if __name__ == "__main__":
    main()
