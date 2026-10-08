"""Fig. 7 (V21): two panels, values read from artifacts only.

(a) Router ablation on the route-budget grid (3 runs x 2 backends x 5 splits x 4 budgets = 120
    cells; D - A on the 62 fair cells). Sources: router_review_analyses_20261004.json,
    router_review_analyses_v2_20261004.json, router_nested_bootstrap_20261004.json.
(b) Routing (D) versus static deployment charged its own latency, per backend, on the
    static-own-cost grid (3 runs x 5 splits x 8 budgets = 120 cells per backend).
    Primary: cells where a route is feasible (filled); diagnostic: all cells including
    route-infeasible budgets where D violates (open). Source: phaseA_static_feasibility_20261004.json.
Intervals: image-level paired bootstrap (1000 reps, seed 0); training-seed variability not included.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pace_style_v13 import FULL_WIDTH_MM, INK, MUTED, apply_style, mm_to_inches, save_all

REP = Path(__file__).parents[3] / "reports"
GEN = Path(__file__).parents[1] / "generated"
DEV = {"E3_logits": ("AGX Xavier", "#4C72B0"), "E1_explicit_float32": ("Hailo-8", "#C44E52")}


def j(n):
    return json.loads((REP / n).read_text())


def main() -> None:
    r = j("router_review_analyses_20261004.json")["median"]
    v2 = j("router_review_analyses_v2_20261004.json")["median"]
    nb = j("router_nested_bootstrap_20261004.json")
    pa = j("phaseA_static_feasibility_20261004.json")
    apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(mm_to_inches(FULL_WIDTH_MM), mm_to_inches(62)),
                               gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.75))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.86, bottom=0.2)

    # (a)
    rows = [("D − A  (fair, n=62)", r["D_vs_A_fair"], nb["D_minus_A_fair_points"]["ci95"]),
            ("A-hard − A  (fair, n=62)", r["A_hard_vs_A_fair"], None),
            ("D − A-hard  (n=120)", r["D_vs_A_hard"], nb["D_minus_A_hard_points"]["ci95"]),
            ("T-hard − A-hard  (n=120)", v2["T_hard_vs_A_hard"], None),
            ("D − T-hard  (n=120)", v2["D_vs_T_hard"], None)]
    y = np.arange(len(rows))[::-1]
    a.axvline(0, color=INK, lw=0.7)
    for yi, (lab, c, nci) in zip(y, rows):
        m = c["mean_delta_points"]
        if "ci95_points" in c:
            a.plot(c["ci95_points"], [yi] * 2, color=INK, lw=1.4)
        if nci:
            a.plot(nci, [yi - 0.22] * 2, color=INK, lw=0.8, ls=(0, (2, 1.2)))
        a.plot(m, yi, "o", ms=4, color=INK)
        a.text(3.6, yi, f"{m:+.2f}", va="center", ha="left", fontsize=6.3)
    a.set_yticks(y, [x[0] for x in rows], fontsize=6.5)
    a.set_xlim(-0.6, 3.6)
    a.set_xlabel("Δ mIoU (points)")
    a.set_title("(a) Router ablation, route-budget grid", fontsize=7.5, loc="left")
    a.plot([], [], color=INK, lw=1.4, label="95% CI (image bootstrap)")
    a.plot([], [], color=INK, lw=0.8, ls=(0, (2, 1.2)), label="nested CI (refit)")
    a.legend(loc="lower right", frameon=False, fontsize=5.8)

    # (b)
    yy, labs = [], []
    k = 0
    for key, (name, col) in DEV.items():
        d = pa[key]
        f, al = d["D_vs_static_route_feasible"], d["D_vs_static_all"]
        for lab, c, filled in ((f"{name}: route-feasible  (n={f['cells']})", f, True),
                               (f"{name}: all cells, diag.  (n={al['cells']})", al, False)):
            yi = -k
            b.plot(c["ci95_points"], [yi] * 2, color=col, lw=1.4 if filled else 0.9)
            b.plot(c["mean_delta_points"], yi, "o", ms=4.5, mfc=col if filled else "white", mec=col, mew=1.0)
            b.text(c["mean_delta_points"] - 0.25, yi + 0.28, f"{c['mean_delta_points']:+.2f}", ha="center",
                   va="bottom", fontsize=6.3, color=col)
            yy.append(yi); labs.append(lab); k += 1
        b.text(-9.3, -k + 0.45, f"route infeasible in {d['cells_static_feasible_route_infeasible']}/120 cells "
               f"({d['share_static_feasible_route_infeasible']:.1%})", fontsize=6, color=MUTED, ha="left", style="italic")
        k += 0.8
    b.axvline(0, color=INK, lw=0.7)
    b.set_yticks(yy, labs, fontsize=6.5)
    b.set_xlim(-9.5, 1.0)
    b.set_ylim(min(yy) - 1.0, 0.8)
    b.set_xlabel("Δ mIoU, D − best feasible static (points)")
    b.set_title("(b) Routing vs static, own-cost grid per device", fontsize=7.5, loc="left")
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(width=0.5, length=2)
    save_all(fig, GEN / "fig7_two_panel_v21")


if __name__ == "__main__":
    main()
