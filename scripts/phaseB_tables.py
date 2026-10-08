"""Write the Phase B robustness tables (supplement) from the Phase B JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

REP = Path("reports")
OUT = Path("paper/submission/ivc_2026-10-09_v21_revision/tables")
ps = json.loads((REP / "phaseB_per_seed_20261004.json").read_text())
sq = json.loads((REP / "phaseB_sequence_split_20261004.json").read_text())
pa = json.loads((REP / "phaseA_static_feasibility_20261004.json").read_text())
r = json.loads((REP / "router_review_analyses_20261004.json").read_text())["median"]
v2 = json.loads((REP / "router_review_analyses_v2_20261004.json").read_text())["median"]


def f(c):
    return f"{c['mean_delta_points']:+.2f} ({c['cells']})"


rows = [("D $-$ A, fair", "D_vs_A_fair", r["D_vs_A_fair"], sq["route_grid"]["D_vs_A_fair"]),
        ("D $-$ A-hard", "D_vs_A_hard", r["D_vs_A_hard"], sq["route_grid"]["D_vs_A_hard"]),
        ("D $-$ T-hard", "D_vs_T_hard", v2["D_vs_T_hard"], sq["route_grid"]["D_vs_T_hard"]),
        ("D $-$ static, route cost", "D_vs_static_route_cost", r["D_vs_static"], sq["route_grid"]["D_vs_static_route_cost"]),
        ("D $-$ static own, AGX, feasible", "D_vs_static_own_E3_logits_feasible",
         pa["E3_logits"]["D_vs_static_route_feasible"], sq["static_own_E3_logits"]["feasible"]),
        ("D $-$ static own, Hailo-8, feasible", "D_vs_static_own_E1_explicit_float32_feasible",
         pa["E1_explicit_float32"]["D_vs_static_route_feasible"], sq["static_own_E1_explicit_float32"]["feasible"])]
t = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
     r"Comparison (median) & Run A & Run B & Run C & Pooled & Sequence split \\", r"\midrule"]
for lab, key, pooled, seq in rows:
    t.append(f"{lab} & {f(ps['a'][key])} & {f(ps['b'][key])} & {f(ps['c'][key])} & {f(pooled)} & {f(seq)} \\\\")
t += [r"\bottomrule", r"\end{tabular}", ""]
(OUT / "robustness_table.tex").write_text("\n".join(t), encoding="utf-8")
print("\n".join(t))
