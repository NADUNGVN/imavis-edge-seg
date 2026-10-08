"""Verify that key numbers printed in the V21 manuscript match the result artifacts.

Each check formats the artifact value exactly as printed and searches for it in main.tex
(and supplement tables). Prints PASS/FAIL per check and exits non-zero on any FAIL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
REP = ROOT / "reports"
DOC = ROOT / "paper/submission/ivc_2026-10-09_v21_revision"
tex = (DOC / "main.tex").read_text(encoding="utf-8")
sup = (DOC / "tables/e1_sensitivity_table.tex").read_text(encoding="utf-8")


def j(n):
    return json.loads((REP / n).read_text())


r = j("router_review_analyses_20261004.json")["median"]
v2 = j("router_review_analyses_v2_20261004.json")["median"]
nb = j("router_nested_bootstrap_20261004.json")
pa = j("phaseA_static_feasibility_20261004.json")
f2 = lambda x: f"{x:+.2f}".replace("-", "$-")
checks = []


def need(label, text, where=tex):
    checks.append((label, text in where, text))


def pm(x):  # LaTeX form used in tables: +2.87 or $-0.03$
    return f"+{x:.2f}" if x >= 0 else f"$-{abs(x):.2f}$"


def ci(c):
    return f"[{pm(c[0])}, {pm(c[1])}]"


need("D-A mean", f"& 62 & 54/6/2 & {pm(r['D_vs_A_fair']['mean_delta_points'])} & {ci(r['D_vs_A_fair']['ci95_points'])}")
need("D-A nested", ci(nb["D_minus_A_fair_points"]["ci95"]))
need("D-A-hard", f"{pm(r['D_vs_A_hard']['mean_delta_points'])} & {ci(r['D_vs_A_hard']['ci95_points'])} & {ci(nb['D_minus_A_hard_points']['ci95'])}")
need("D-T-hard", f"{pm(v2['D_vs_T_hard']['mean_delta_points'])} & {ci(v2['D_vs_T_hard']['ci95_points'])}")
need("D-static route cost", f"{pm(r['D_vs_static']['mean_delta_points'])} & {ci(r['D_vs_static']['ci95_points'])}")
need("violations A", f"A {r['violating_cells']['A']} / 58 / 59 of 120")
for key, name in (("E3_logits", "AGX Xavier"), ("E1_explicit_float32", "Hailo-8")):
    d = pa[key]
    f, a = d["D_vs_static_route_feasible"], d["D_vs_static_all"]
    need(f"{name} feasible row", f"{name} & D $-$ static, route-feasible (primary) & {f['cells']} & {f['wins']}/{f['ties']}/{f['losses']} & {pm(f['mean_delta_points'])} & {ci(f['ci95_points'])}")
    need(f"{name} all row", f"{name} & D $-$ static, all cells (diagnostic) & {a['cells']} & {a['wins']}/{a['ties']}/{a['losses']} & {pm(a['mean_delta_points'])} & {ci(a['ci95_points'])}")
    n = d["cells_static_feasible_route_infeasible"]
    need(f"{name} infeasible count", f"{n}/120 & \\multicolumn{{4}}{{l}}{{{d['share_static_feasible_route_infeasible'] * 100:.1f}\\% of the grid")
    need(f"{name} D violations == infeasible", "", "") if d["D_violations_all"] == n and d["D_violations_feasible"] == 0 else checks.append((f"{name} D violations", False, str(d["D_violations_all"])))
    need(f"{name} supplement row", f"{f['mean_delta_points']:+.2f} [{f['ci95_points'][0]:+.2f}, {f['ci95_points'][1]:+.2f}] ({f['cells']})", sup)
need("abstract feasible numbers", "by 3.0 points on a TensorRT GPU and 8.2 points on a Hailo-8")
need("abstract shares", "at 12.5\\% and 37.5\\% of the evaluated budget cells")
need("results AGX feasible text", "beats D by 3.00 points on AGX Xavier (95\\% CI $-3.07$ to $-2.72$; 0 wins, 46 ties, 59 losses over 105 cells)")
need("results Hailo feasible text", "8.18 points on Hailo-8 with FLOAT32 output streams ($-8.42$ to $-7.40$; 0/6/69 over 75 cells)")
need("UINT8 feasible", f"{abs(pa['E1_explicit_uint8']['D_vs_static_route_feasible']['mean_delta_points']):.2f} points with UINT8")
need("T-hard feasible", f"($-{abs(pa['E3_logits']['T_hard_vs_static_route_feasible']['mean_delta_points']):.2f}$ and $-{abs(pa['E1_explicit_float32']['T_hard_vs_static_route_feasible']['mean_delta_points']):.2f}$)")
for bad in ("adds nothing", "matched candidate-specific calibration", "statistically indistinguishable", "D, pooled D, static, and LOCO D 0 of 120 in every table",
            "dominate the short tiny inference", "engine switch that dominates"):
    checks.append((f"absent: {bad}", bad not in tex, bad))

fails = 0
for label, ok, text in checks:
    print(("PASS" if ok else "FAIL"), label, "" if ok else f"-> {text}")
    fails += not ok
print(f"{len(checks) - fails}/{len(checks)} checks passed")
sys.exit(1 if fails else 0)
