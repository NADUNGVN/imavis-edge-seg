"""Phase A (V21 revision): static-own-cost comparison with explicit route feasibility.

For each same-harness device configuration the budget grid is the union of the four
complete-route costs C_h and the four static costs S_h (8 budgets). A budget is
route-infeasible when it is below min_l C_h(l) = C_h(tiny): every route starts with the
tiny probe, so no routed policy can meet it, while the static tiny engine (S_h(tiny) <
C_h(tiny)) can. In such cells run_policy/d_choice falls back to tiny and records a
violation.

Reports, per configuration and policy (D, T-hard):
  all cells (as in V20 Table 7, recomputed here with a fresh bootstrap),
  route-feasible cells only (conditional comparison, fresh bootstrap),
  number and share of cells where static is feasible but routing is not.
Image-level bootstrap only (1000 reps, held-out images resampled per split, shared
across runs and budgets); it does NOT include training-seed variability.
Output: reports/phaseA_static_feasibility_20261004.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from router_review_analyses import BASE, RUNS, Split, run_policy
from router_review_analyses_v2 import bootstrap, compare, static_fair, threshold_policy
from router_same_harness_analysis import configurations

REPS = 1000


def main() -> None:
    splits, grids = {}, {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        ev = json.loads((BASE / f"run_{run}_evaluation.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
        grids[run] = {k: ev[k]["risk_target_grid"] for k in splits[run]}
    out = {"_definition": __doc__}
    for name, (rc, sc) in configurations("median").items():
        budgets = sorted(set(rc.values()) | set(sc.values()))
        cheapest_route = min(rc.values())
        rows = []
        for run in RUNS:
            for s, sp in splits[run].items():
                D = run_policy("D", sp, grids[run][s], budgets, rc, sp.cal)
                TH = threshold_policy(sp, budgets, rc)
                ST = static_fair(sp, budgets, sc)
                for b in budgets:
                    rows.append({"run": run, "split": s, "budget": b, "route_feasible": b >= cheapest_route - 1e-9,
                                 "static_feasible": ST[b] is not None, "D": D[b], "T_hard": TH[b], "static_S": ST[b]})
        feas = [r for r in rows if r["route_feasible"]]
        infeas = [r for r in rows if not r["route_feasible"]]
        block = {
            "route_ms": rc, "static_ms": sc, "budgets_ms": budgets, "cheapest_route_ms": cheapest_route,
            "grid": f"{len(RUNS)} runs x 5 splits x {len(budgets)} budgets = {len(rows)} cells",
            "route_infeasible_budgets_ms": [b for b in budgets if b < cheapest_route - 1e-9],
            "cells_static_feasible_route_infeasible": sum(r["static_feasible"] and not r["route_feasible"] for r in rows),
            "share_static_feasible_route_infeasible": sum(r["static_feasible"] and not r["route_feasible"] for r in rows) / len(rows),
            "D_violations_all": int(sum(r["D"]["violation"] > 0 for r in rows)),
            "D_violations_feasible": int(sum(r["D"]["violation"] > 0 for r in feas)),
            "T_hard_violations_all": int(sum(r["T_hard"]["violation"] > 0 for r in rows)),
        }
        for pol in ("D", "T_hard"):
            a = compare(rows, pol, "static_S"); a["ci95_points"] = bootstrap(rows, pol, "static_S", reps=REPS)
            f = compare(feas, pol, "static_S"); f["ci95_points"] = bootstrap(feas, pol, "static_S", reps=REPS)
            i = compare(infeas, pol, "static_S") if infeas else None
            block[f"{pol}_vs_static_all"] = a
            block[f"{pol}_vs_static_route_feasible"] = f
            block[f"{pol}_vs_static_route_infeasible"] = i
        out[name] = block
        print(name, block["grid"], "infeasible budgets", block["route_infeasible_budgets_ms"],
              "| all", round(block["D_vs_static_all"]["mean_delta_points"], 2), block["D_vs_static_all"]["ci95_points"],
              "| feasible", block["D_vs_static_route_feasible"]["cells"], round(block["D_vs_static_route_feasible"]["mean_delta_points"], 2),
              block["D_vs_static_route_feasible"]["ci95_points"], flush=True)
    Path("reports/phaseA_static_feasibility_20261004.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
