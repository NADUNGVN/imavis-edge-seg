"""Audit (read-only): per-budget breakdown of the same-harness D-vs-static comparison,
flagging budgets below the cheapest route (no feasible route; D falls back to tiny and
violates). Writes reports/audit_static_cells_20261004.json. No new measurements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from router_review_analyses import BASE, RUNS, Split, run_policy
from router_review_analyses_v2 import static_fair
from router_same_harness_analysis import configurations

out = {}
splits, grids = {}, {}
for run in RUNS:
    dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
    ev = json.loads((BASE / f"run_{run}_evaluation.json").read_text())
    splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
    grids[run] = {k: ev[k]["risk_target_grid"] for k in splits[run]}
for name, (rc, sc) in configurations("median").items():
    if name not in ("E3_logits", "E1_explicit_float32"):
        continue
    budgets = sorted(set(rc.values()) | set(sc.values()))
    per_b = {b: [] for b in budgets}
    for run in RUNS:
        for s, sp in splits[run].items():
            D = run_policy("D", sp, grids[run][s], budgets, rc, sp.cal)
            ST = static_fair(sp, budgets, sc)
            for b in budgets:
                per_b[b].append((D[b]["miou"] - ST[b]["miou"]) * 100 > 0 and 1 or
                                (abs(D[b]["miou"] - ST[b]["miou"]) * 100 <= 1e-10 and 0 or -1))
                per_b[b][-1] = (per_b[b][-1], float(D[b]["violation"] > 0), (D[b]["miou"] - ST[b]["miou"]) * 100)
    rows, feas = [], []
    for b in budgets:
        v = per_b[b]
        no_route = b < min(rc.values()) - 1e-9
        rec = {"budget_ms": round(b, 2), "no_feasible_route": no_route, "cells": len(v),
               "wins": sum(x[0] == 1 for x in v), "ties": sum(x[0] == 0 for x in v), "losses": sum(x[0] == -1 for x in v),
               "D_violating_cells": int(sum(x[1] for x in v)), "mean_delta_points": float(np.mean([x[2] for x in v]))}
        rows.append(rec)
        if not no_route:
            feas += [x[2] for x in v]
    out[name] = {"per_budget": rows,
                 "feasible_only": {"cells": len(feas), "mean_delta_points": float(np.mean(feas)),
                                   "wins": int(sum(d > 1e-10 for d in feas)), "ties": int(sum(abs(d) <= 1e-10 for d in feas)),
                                   "losses": int(sum(d < -1e-10 for d in feas))}}
Path("reports/audit_static_cells_20261004.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
