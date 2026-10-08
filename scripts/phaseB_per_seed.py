"""Phase B robustness: main comparisons split by training run (seed 0/1/2 = Run A/B/C).

Same operating-point selection as the pooled analyses; each run's cells are summarized
separately (no bootstrap; the spread across runs is the quantity of interest).
Output: reports/phaseB_per_seed_20261004.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from router_review_analyses import BASE, RUNS, Split, build_cells, compare
from router_review_analyses_v2 import compare as compare2, static_fair, threshold_policy
from router_same_harness_analysis import configurations
from router_review_analyses import run_policy


def load():
    splits, grids = {}, {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        ev = json.loads((BASE / f"run_{run}_evaluation.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
        grids[run] = {k: ev[k]["risk_target_grid"] for k in splits[run]}
    return splits, grids


def main() -> None:
    splits, grids = load()
    out = {"_note": "per-run summaries; Run A/B/C = seeds 0/1/2"}
    cells = build_cells(splits, grids, "median")
    for run in RUNS:
        rc = [c for c in cells if c["run"] == run]
        # T-hard on route grid
        out[run] = {
            "D_vs_A_fair": compare(rc, "D", "A", "A"),
            "D_vs_A_hard": compare(rc, "D", "A_hard", None),
            "D_vs_static_route_cost": compare(rc, "D", "static", None),
        }
    # D vs T-hard and static own cost, per run
    for name, (rcost, scost) in configurations("median").items():
        if name not in ("E3_logits", "E1_explicit_float32"):
            continue
        budgets = sorted(set(rcost.values()) | set(scost.values()))
        cheapest = min(rcost.values())
        for run in RUNS:
            rows = []
            for s, sp in splits[run].items():
                D = run_policy("D", sp, grids[run][s], budgets, rcost, sp.cal)
                ST = static_fair(sp, budgets, scost)
                for b in budgets:
                    rows.append({"run": run, "split": s, "budget": b, "D": D[b], "static_S": ST[b], "feas": b >= cheapest - 1e-9})
            out[run][f"D_vs_static_own_{name}_feasible"] = compare2([r for r in rows if r["feas"]], "D", "static_S")
            out[run][f"D_vs_static_own_{name}_all"] = compare2(rows, "D", "static_S")
    # D vs T-hard per run on the route grid (costs = route costs per backend)
    from router_review_analyses import load_costs
    for run in RUNS:
        rows = []
        for backend, cost in load_costs("median").items():
            budgets = sorted(set(cost.values()))
            for s, sp in splits[run].items():
                D = run_policy("D", sp, grids[run][s], budgets, cost, sp.cal)
                TH = threshold_policy(sp, budgets, cost)
                for b in budgets:
                    rows.append({"split": s, "D": D[b], "T_hard": TH[b]})
        out[run]["D_vs_T_hard"] = compare2(rows, "D", "T_hard")
    Path("reports/phaseB_per_seed_20261004.json").write_text(json.dumps(out, indent=2, default=float))
    for run in RUNS:
        print(run, {k: (v["cells"], round(v["mean_delta_points"], 2)) for k, v in out[run].items()})


if __name__ == "__main__":
    main()
