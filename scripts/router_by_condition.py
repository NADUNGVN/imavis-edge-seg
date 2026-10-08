"""Condition-level breakdown of D / pooled D against rank escalation A (median cost table).

Reuses router_review_analyses.build_cells, so operating points are chosen exactly as in
the pooled analysis. Writes reports/router_by_condition_<TAG>.json and a LaTeX tabular
for the manuscript (Table "router-condition").

Landscape rerun:
  PACE_ROUTER_BASE=reports/landscape_20261004 PACE_TAG=20261004 PACE_ROUTE_COSTS=same_harness \
      python scripts/router_by_condition.py --tex paper/.../tables/router_by_condition_table_points.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from router_review_analyses import BASE, RUNS, TAG, Split, build_cells, compare

NAMES = {"cityscapes": "Cityscapes", "acdc/fog": "ACDC/Fog", "acdc/night": "ACDC/Night",
         "acdc/rain": "ACDC/Rain", "acdc/snow": "ACDC/Snow"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tex", type=Path, default=None)
    args = p.parse_args()
    splits, grids = {}, {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        ev = json.loads((BASE / f"run_{run}_evaluation.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
        grids[run] = {k: ev[k]["risk_target_grid"] for k in splits[run]}
    cells = build_cells(splits, grids, "median")
    out = {}
    for s in NAMES:
        rows = [c for c in cells if c["split"] == s]
        out[s] = {
            "cells": len(rows),
            "D_vs_A_fair": compare(rows, "D", "A", "A"),
            "D_pooled_vs_A_fair": compare(rows, "D_pooled", "A", "A"),
            "D_vs_A_hard": compare(rows, "D", "A_hard", None),
            "A_violating": int(sum(c["A"]["violation"] > 0 for c in rows)),
            "D_violating": int(sum(c["D"]["violation"] > 0 for c in rows)),
        }
    Path(f"reports/router_by_condition_{TAG}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    if args.tex:
        lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
                 r"Condition & Fair & Config. W/T/L & Config. $\Delta$ & Pooled W/T/L & Pooled $\Delta$ & A viol. & D viol. \\",
                 r"\midrule"]
        for s, name in NAMES.items():
            o = out[s]
            d, q = o["D_vs_A_fair"], o["D_pooled_vs_A_fair"]
            fmt = lambda x: "--" if x is None else f"{x:+.2f}"
            lines.append(f"{name} & {d['cells']}/{o['cells']} & {d['wins']}/{d['ties']}/{d['losses']} & {fmt(d['mean_delta_points'])} & "
                         f"{q['wins']}/{q['ties']}/{q['losses']} & {fmt(q['mean_delta_points'])} & {o['A_violating']}/{o['cells']} & "
                         f"{o['D_violating']}/{o['cells']} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        args.tex.write_text("\n".join(lines))
        print(f"wrote {args.tex}")


if __name__ == "__main__":
    main()
