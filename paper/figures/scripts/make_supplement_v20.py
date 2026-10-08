"""V20 supplement assets from recorded landscape results (no new experiments).

  tables/compiled_engine_table.tex   compiled TensorRT FP16 / Hailo-8 INT8 vs PyTorch (Run A)
  tables/e1_sensitivity_table.tex    Hailo-8 activation path / output stream sensitivity
  figures/figS2_route_cost_expansion_v20.pdf  candidate-only p95 vs complete-route median
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pace_style_v13 import CANDIDATE_COLORS, FULL_WIDTH_MM, INK, apply_style, mm_to_inches, save_all

ROOT = Path(__file__).parents[3]
REP = ROOT / "reports"
SUP = ROOT / "paper/submission/ivc_2026-10-09_v21_revision"
LEVELS = ("tiny", "small", "medium", "large")
SPLITS = [("cityscapes", "Cityscapes"), ("acdc_fog", "Fog"), ("acdc_night", "Night"), ("acdc_rain", "Rain"),
          ("acdc_snow", "Snow")]


def compiled_table() -> None:
    rows = []
    for f, name in (("compiled_eval_E3_20261004.json", "AGX Xavier, TensorRT FP16"),
                    ("compiled_eval_E1_20261004.json", "Hailo-8, INT8")):
        r = json.loads((REP / f).read_text())["results"]
        for lv in LEVELS:
            d = [r[f"{s}|{lv}"]["delta_points"] for s, _ in SPLITS]
            a = [r[f"{s}|{lv}"]["pixel_agreement"] * 100 for s, _ in SPLITS]
            rows.append(f"{name} & {lv} & " + " & ".join(f"{x:+.2f}" for x in d) + f" & {min(a):.1f} \\\\")
        rows.append(r"\midrule")
    rows.pop()
    tex = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
           r"Backend & Capacity & " + " & ".join(n for _, n in SPLITS) + r" & Min.\ agreement (\%) \\", r"\midrule",
           *rows, r"\bottomrule", r"\end{tabular}", ""]
    (SUP / "tables/compiled_engine_table.tex").write_text("\n".join(tex), encoding="utf-8")


def e1_table() -> None:
    sh = json.loads((REP / "router_same_harness_analysis_20261004.json").read_text())["median"]
    pa = json.loads((REP / "phaseA_static_feasibility_20261004.json").read_text())
    names = {"E3_logits": "AGX Xavier (logits)", "E1_explicit_float32": "Hailo-8 explicit, FLOAT32 (primary)",
             "E1_explicit_uint8": "Hailo-8 explicit, UINT8", "E1_scheduler_float32": "Hailo-8 scheduler, FLOAT32",
             "E1_scheduler_uint8": "Hailo-8 scheduler, UINT8"}
    out = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
           r"Configuration & Overhead $C-S$ (ms) & D $-$ static, feasible [95\% CI] (n) & D $-$ static, all (diag.) & Route-infeasible cells & Mixture gain \\",
           r"\midrule"]
    for k, n in names.items():
        v, a = sh[k], pa[k]
        ov = [v["route_ms"][lv] - v["static_ms"][lv] for lv in LEVELS]
        f, al = a["D_vs_static_route_feasible"], a["D_vs_static_all"]
        out.append(f"{n} & {min(ov):.1f}--{max(ov):.1f} & {f['mean_delta_points']:+.2f} [{f['ci95_points'][0]:+.2f}, {f['ci95_points'][1]:+.2f}] ({f['cells']}) & "
                   f"{al['mean_delta_points']:+.2f} & {a['cells_static_feasible_route_infeasible']}/120 & "
                   f"{v['mixture_vs_static']['mean_gain_points']:+.2f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    (SUP / "tables/e1_sensitivity_table.tex").write_text("\n".join(out), encoding="utf-8")


def route_expansion() -> None:
    cand = {r["candidate"]: r for r in csv.DictReader(open(ROOT / "paper/tables/candidate_family_landscape.csv", encoding="utf-8"))}
    sh = json.loads((REP / "router_same_harness_analysis_20261004.json").read_text())["median"]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_inches(FULL_WIDTH_MM), mm_to_inches(58)))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.86, wspace=0.22)
    for ax, (key, col, title) in zip(axes, (("E3_logits", "e3_candidate_p95_ms", "AGX Xavier (TensorRT FP16)"),
                                            ("E1_explicit_float32", "e1_candidate_p95_ms", "Hailo-8 (INT8)"))):
        x = np.arange(4)
        c = np.array([float(cand[lv][col]) for lv in LEVELS])
        r = np.array([sh[key]["route_ms"][lv] for lv in LEVELS])
        cols = [CANDIDATE_COLORS[lv] for lv in LEVELS]
        ax.bar(x - 0.19, c, 0.36, color=cols, alpha=0.5, ec=INK, lw=0.5)
        ax.bar(x + 0.19, r, 0.36, color=cols, ec=INK, lw=0.5)
        for i in range(4):
            ax.text(x[i] + 0.19, r[i], f"{r[i] / c[i]:.1f}×", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x, [lv.title() for lv in LEVELS])
        ax.set_ylabel("latency (ms)")
        ax.set_title(title, fontsize=7.5)
        ax.set_ylim(0, r.max() * 1.15)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#BBBBBB", alpha=0.5, ec=INK, lw=0.5),
                            plt.Rectangle((0, 0), 1, 1, fc="#777777", ec=INK, lw=0.5)],
                   labels=["candidate-only p95 (left)", "complete-route median (right)"], loc="upper center",
                   bbox_to_anchor=(1.1, 1.22), ncol=2, frameon=False, fontsize=6.6)
    save_all(fig, SUP / "figures/figS2_route_cost_expansion_v20")


if __name__ == "__main__":
    compiled_table()
    e1_table()
    route_expansion()
    print("ok")
