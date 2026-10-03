"""Static vs routing with SAME-HARNESS cost tables (review round 2, 2026-10-03).

Inputs: reports/static_vs_route_E3_20261003.json and the E1 files
reports/static_vs_route_E1_<path>_<format>_20261003.json produced by
scripts/measure_static_vs_route_{trt,hailo}.py. For every device configuration the
route table C_h and the static table S_h come from the SAME file, so they share the
harness, inputs and timer boundary. Routing policies (D, T-hard, A-hard) use C_h;
the best feasible static candidate and static mixing use S_h. Budgets are the
union of both tables' values for the configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from router_review_analyses import BASE, LEVELS, RUNS, Split, run_policy
from router_review_analyses_v2 import bootstrap, compare, mixture_gain, static_fair, threshold_policy

STAT_KEY = {"median": "median_ms", "p95": "p95_ms", "p99": "p99_ms"}


def tables_from(path: Path, stat: str, suffix: str = "") -> tuple[dict[str, float], dict[str, float]] | None:
    d = json.loads(path.read_text())
    res = d.get("results", {})
    try:
        route = {lv: res[f"route_tiny->{lv}{suffix}"][STAT_KEY[stat]] for lv in LEVELS}
        static = {lv: res[f"static_{lv}{suffix}"][STAT_KEY[stat]] for lv in LEVELS}
    except KeyError:
        return None
    return route, static


def configurations(stat: str) -> dict[str, tuple[dict, dict]]:
    out = {}
    e3 = Path("reports/static_vs_route_E3_20261003.json")
    for mode in ("logits", "none"):
        t = tables_from(e3, stat, f"|{mode}")
        if t:
            out[f"E3_{mode}"] = t
    for path in ("explicit", "scheduler"):
        for fmt in ("float32", "uint8"):
            p = Path(f"reports/static_vs_route_E1_{path}_{fmt}_20261003.json")
            if p.exists():
                t = tables_from(p, stat)
                if t:
                    out[f"E1_{path}_{fmt}"] = t
    return out


def main() -> None:
    splits, grids = {}, {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        ev = json.loads((BASE / f"run_{run}_evaluation.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
        grids[run] = {k: ev[k]["risk_target_grid"] for k in splits[run]}
    out: dict[str, Any] = {}
    for stat in ("median", "p99"):
        confs = configurations(stat)
        out[stat] = {}
        for name, (rc, sc) in confs.items():
            rows = []
            budgets = sorted(set(rc.values()) | set(sc.values()))
            for run in RUNS:
                for split_name, sp in splits[run].items():
                    t = grids[run][split_name]
                    D = run_policy("D", sp, t, budgets, rc, sp.cal)
                    AH = run_policy("A_hard", sp, t, budgets, rc, sp.cal)
                    TH = threshold_policy(sp, budgets, rc)
                    ST = static_fair(sp, budgets, sc)
                    for b in budgets:
                        rows.append({"run": run, "split": split_name, "budget": b,
                                     "D": D[b], "A_hard": AH[b], "T_hard": TH[b], "static_S": ST[b]})
            block = {
                "route_ms": rc, "static_ms": sc, "budgets_ms": budgets,
                "D_vs_static": compare(rows, "D", "static_S"),
                "T_hard_vs_static": compare(rows, "T_hard", "static_S"),
                "D_vs_T_hard": compare(rows, "D", "T_hard"),
                "D_vs_A_hard": compare(rows, "D", "A_hard"),
                "mixture_vs_static": mixture_gain(splits, {name: rc}, {name: sc}),
            }
            if stat == "median":
                block["D_vs_static"]["ci95_points"] = bootstrap(rows, "D", "static_S", reps=500)
            out[stat][name] = block
            print(stat, name, json.dumps({k: block[k] for k in ("D_vs_static", "T_hard_vs_static", "D_vs_T_hard")}),
                  "mix", round(block["mixture_vs_static"]["mean_gain_points"], 2),
                  round(block["mixture_vs_static"]["share_above"], 2), flush=True)
    Path("reports/router_same_harness_analysis_20261003.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/router_same_harness_analysis_20261003.json")


if __name__ == "__main__":
    main()
