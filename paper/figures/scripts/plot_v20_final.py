"""Final V20 figures drawn from recorded results only (no new experiments).

fig7_forest_v20        : router ablation / static comparison forest plot (mean delta, 95% CI)
fig6_rq2_dumbbell_v20  : Elastic Large (shared) vs PACE-Large (alone) per split, delta labelled;
                         Fast-SCNN as a secondary reference
fig2_pipeline_v20      : inference pipeline; the tiny probe always runs, tiny output reused if
                         tiny is selected, one additional engine otherwise

Inputs: reports/router_review_analyses_20261004.json, router_review_analyses_v2_20261004.json,
router_nested_bootstrap_20261004.json, router_same_harness_analysis_20261004.json,
reports/landscape_20261004/eval_*_seed{0,1,2}.json, reports/qualitative_assets_v20/.
Output: paper/figures/generated/*.pdf|svg|png (vector text, pdf.fonttype 42)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from pace_style_v13 import (CANDIDATE_COLORS, CANDIDATES, FULL_WIDTH_MM, HARDWARE, INK, MUTED, PANEL_EDGE, PANEL_FILL,
                            RISK, SINGLE_WIDTH_MM, apply_style, mm_to_inches, save_all)

ROOT = Path(__file__).parents[3]
REP = ROOT / "reports"
GEN = Path(__file__).parents[1] / "generated"
SPLITS = ("cityscapes", "acdc/fog", "acdc/night", "acdc/rain", "acdc/snow")
SPLIT_LABEL = {"cityscapes": "Cityscapes", "acdc/fog": "ACDC fog", "acdc/night": "ACDC night",
               "acdc/rain": "ACDC rain", "acdc/snow": "ACDC snow"}
DEVICE_COLOR = {"E3": "#4C72B0", "E1": "#C44E52"}


def j(name: str) -> dict:
    return json.loads((REP / name).read_text())


# ---------------------------------------------------------------- forest
def fig_forest() -> None:
    r = j("router_review_analyses_20261004.json")["median"]
    v2 = j("router_review_analyses_v2_20261004.json")["median"]
    nb = j("router_nested_bootstrap_20261004.json")
    sh = j("router_same_harness_analysis_20261004.json")["median"]
    rows = [  # label, mean, ci, nested ci, group, cells
        ("D − A  (fair cells)", r["D_vs_A_fair"]["mean_delta_points"], r["D_vs_A_fair"]["ci95_points"],
         nb["D_minus_A_fair_points"]["ci95"], "ingredient", f"{r['D_vs_A_fair']['cells']}"),
        ("D − A-hard", r["D_vs_A_hard"]["mean_delta_points"], r["D_vs_A_hard"]["ci95_points"],
         nb["D_minus_A_hard_points"]["ci95"], "ingredient", "120"),
        ("D − T-hard", v2["D_vs_T_hard"]["mean_delta_points"], v2["D_vs_T_hard"]["ci95_points"], None, "ingredient", "120"),
        ("D − static, charged route cost", r["D_vs_static"]["mean_delta_points"], r["D_vs_static"]["ci95_points"], None,
         "static", "120"),
        ("D − static, own cost, AGX Xavier", sh["E3_logits"]["D_vs_static"]["mean_delta_points"],
         sh["E3_logits"]["D_vs_static"]["ci95_points"], None, "E3", "120"),
        ("D − static, own cost, Hailo-8", sh["E1_explicit_float32"]["D_vs_static"]["mean_delta_points"],
         sh["E1_explicit_float32"]["D_vs_static"]["ci95_points"], None, "E1", "120"),
    ]
    apply_style()
    fig, ax = plt.subplots(figsize=(mm_to_inches(SINGLE_WIDTH_MM), mm_to_inches(62)))
    fig.subplots_adjust(left=0.47, right=0.97, top=0.93, bottom=0.17)
    y = np.arange(len(rows))[::-1].astype(float)
    y[3:] -= 0.5  # gap between the two groups
    ax.axvspan(0, 4.5, color="#E8F1FB", zorder=0)
    ax.axvspan(-9, 0, color="#FBEDEA", zorder=0)
    ax.axvline(0, color=INK, lw=0.7)
    for yi, (lab, m, ci, nci, grp, n) in zip(y, rows):
        col = {"ingredient": INK, "static": MUTED, "E3": DEVICE_COLOR["E3"], "E1": DEVICE_COLOR["E1"]}[grp]
        if nci is not None:
            ax.plot(nci, [yi - 0.18] * 2, color=col, lw=0.8, ls=(0, (2, 1.2)))
        ax.plot(ci, [yi] * 2, color=col, lw=1.4, solid_capstyle="butt")
        ax.plot(m, yi, "o", ms=4, color=col, mec="white", mew=0.5, zorder=3)
        ax.text(max(ci[1], m) + 0.25, yi, f"{m:+.2f}", ha="left", va="center", fontsize=6, color=col)
    ax.set_yticks(y, [row[0] for row in rows], fontsize=6.6)
    ax.set_xlim(-8.2, 4.6)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_xlabel("Δ mIoU (points), mean and 95% CI")
    ax.text(0.15, y.min() - 0.6, "routing better →", fontsize=6, color="#1F6FB2", va="bottom")
    ax.text(-0.15, y.min() - 0.6, "← reference better", fontsize=6, color="#B4442C", va="bottom", ha="right")
    ax.plot([], [], color=INK, lw=0.8, ls=(0, (2, 1.2)), label="nested CI (refit)")
    ax.legend(loc="upper left", frameon=False, fontsize=6, handlelength=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save_all(fig, GEN / "fig7_forest_v20")


# ---------------------------------------------------------------- RQ2 dumbbell
def load_miou(name: str, level: str | None = None) -> np.ndarray:
    out = []
    for s in range(3):
        d = json.loads((REP / "landscape_20261004" / f"eval_{name}_seed{s}.json").read_text())
        d = d[level] if level else d
        out.append([d[sp]["miou"] * 100 for sp in SPLITS])
    return np.array(out)


def fig_rq2() -> None:
    el, al, fs = load_miou("supernet", "large"), load_miou("pace_large"), load_miou("fast_scnn")
    apply_style()
    fig, ax = plt.subplots(figsize=(mm_to_inches(SINGLE_WIDTH_MM), mm_to_inches(60)))
    fig.subplots_adjust(left=0.2, right=0.83, top=0.80, bottom=0.17)
    y = np.arange(len(SPLITS))[::-1]
    c_el, c_al, c_fs = CANDIDATE_COLORS["large"], "#7F7F7F", "#BBBBBB"
    for yi, a, b, c in zip(y, el.mean(0), al.mean(0), fs.mean(0)):
        ax.plot([a, b], [yi, yi], color=INK, lw=0.8, zorder=1)
        ax.text(1.02, yi, f"{a - b:+.1f}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.6,
                color=INK, fontweight="bold")
    ax.errorbar(fs.mean(0), y - 0.22, xerr=fs.std(0), fmt="d", ms=3.2, color=c_fs, mec="#888888", mew=0.4,
                elinewidth=0.6, capsize=1.2, label="Fast-SCNN (reference)", zorder=2)
    ax.errorbar(al.mean(0), y, xerr=al.std(0), fmt="o", ms=4.2, color=c_al, elinewidth=0.6, capsize=1.2,
                label="PACE-Large, trained alone", zorder=3)
    ax.errorbar(el.mean(0), y, xerr=el.std(0), fmt="o", ms=4.2, color=c_el, elinewidth=0.6, capsize=1.2,
                label="Elastic Large, shared", zorder=4)
    ax.text(1.02, y.max() + 0.75, "Δ shared\n− alone", transform=ax.get_yaxis_transform(), fontsize=6, color=MUTED,
            va="center")
    ax.set_yticks(y, [SPLIT_LABEL[s] for s in SPLITS])
    ax.set_xlabel("mIoU (%), mean ± s.d. over 3 seeds")
    ax.set_ylim(-0.6, len(SPLITS) - 0.4)
    ax.grid(axis="x", color="#E5E7EB", lw=0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.25, 1.06), ncol=2, frameon=False, fontsize=5.6,
              handletextpad=0.2, columnspacing=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save_all(fig, GEN / "fig6_rq2_dumbbell_v20")


# ---------------------------------------------------------------- pipeline
def box(ax, x, y, w, h, title, sub=None, fc="white", ec=PANEL_EDGE, tc=INK, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h * 0.62 if sub else y + h - 0.04, title, ha="center", va="center", fontsize=6.8,
            fontweight="bold", color=tc)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=5.6, color=MUTED, linespacing=1.15)


def arrow(ax, a, b, color=INK, ls="-", lw=0.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


def fig_pipeline() -> None:
    apply_style()
    A = REP / "qualitative_assets_v20"
    aid = "acdc-rain_GP020402_frame_000863_rgb_anon"
    W, H = mm_to_inches(FULL_WIDTH_MM), mm_to_inches(62)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_aspect("auto")
    # lanes
    ax.add_patch(Rectangle((0.005, 0.50), 0.99, 0.48, fc="#FDF2F8", ec="none"))
    ax.add_patch(Rectangle((0.005, 0.03), 0.99, 0.43, fc=PANEL_FILL, ec="none"))
    ax.text(0.012, 0.955, "VISUAL RISK  (per image, always executed)", fontsize=6, color=RISK, fontweight="bold")
    ax.text(0.012, 0.43, "HARDWARE COST  (per device, measured once)", fontsize=6, color=HARDWARE, fontweight="bold")

    def thumb(path, x, y, w):
        im = Image.open(path).convert("RGB").resize((256, 128))
        h = w * 0.5 * W / H
        ax.imshow(np.asarray(im), extent=(x, x + w, y, y + h), zorder=3, aspect="auto")
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.4, zorder=4))
        return h

    from render_palette import colorize_png  # noqa: E402  (local helper)
    hh = thumb(A / f"{aid}_rgb.jpg", 0.02, 0.62, 0.10)
    ax.text(0.07, 0.62 + hh + 0.02, "Input (pre-processed tensor)", ha="center", fontsize=6)
    colorize_png(A / f"{aid}_tiny.png", GEN / "_tmp_tiny.png")
    thumb(GEN / "_tmp_tiny.png", 0.165, 0.62, 0.10)
    ax.text(0.215, 0.62 + hh + 0.02, "1  Tiny engine (probe)", ha="center", fontsize=6, fontweight="bold")
    ax.text(0.215, 0.575, "segmentation kept in memory", ha="center", fontsize=5.4, color=MUTED)
    arrow(ax, (0.122, 0.62 + hh / 2), (0.163, 0.62 + hh / 2))
    box(ax, 0.305, 0.63, 0.085, 0.16, "2  Entropy", "mean pixel\nentropy $s(x)$", ec=RISK)
    arrow(ax, (0.267, 0.62 + hh / 2), (0.303, 0.71))
    box(ax, 0.42, 0.58, 0.15, 0.26, "3  Calibrated risk", None, ec=RISK)
    for k, lv in enumerate(CANDIDATES):
        yy = 0.75 - k * 0.045
        ax.plot([0.43, 0.455], [yy, yy], color=CANDIDATE_COLORS[lv], lw=2)
        ax.text(0.46, yy, f"$\\hat r_{{{lv[0].upper()}}} = g_{{{lv[0].upper()}}}(s(x))$", va="center", fontsize=5.8)
    arrow(ax, (0.39, 0.71), (0.418, 0.71))
    # hardware lane
    box(ax, 0.06, 0.12, 0.17, 0.20, "Target device $h$", "compile 4 static engines;\nmeasure complete routes\nand static deployments")
    box(ax, 0.30, 0.08, 0.27, 0.28, "Measured route cost $C_h(\\ell)$", None, ec=HARDWARE)
    for k, lv in enumerate(CANDIDATES):
        xx, yy = 0.315 + (k % 2) * 0.13, 0.25 - (k // 2) * 0.06
        ax.plot([xx, xx + 0.025], [yy, yy], color=CANDIDATE_COLORS[lv], lw=2)
        ax.text(xx + 0.03, yy, f"$C_h$({lv.title()})", va="center", fontsize=5.8)
    ax.plot([0.315, 0.34], [0.115, 0.115], color=INK, lw=0.8, ls=(0, (2, 1.5)))
    ax.text(0.345, 0.115, "budget $B_h$; preprocessed tensor → synchronized output", va="center", fontsize=5.4)
    arrow(ax, (0.23, 0.22), (0.298, 0.22))
    # policy
    box(ax, 0.615, 0.30, 0.13, 0.40, "4  Policy", None, fc="#F0FDF4", ec="#16A34A")
    ax.text(0.622, 0.58, "1. feasible: $C_h(\\ell)\\leq B_h$\n2. cheapest with $\\hat r_\\ell\\leq\\tau$\n3. else lowest $\\hat r_\\ell$",
            fontsize=5.6, va="top", linespacing=1.4)
    ax.text(0.68, 0.33, "D (or threshold\nrule T-hard)", ha="center", fontsize=5.4, color=MUTED)
    arrow(ax, (0.57, 0.71), (0.613, 0.62))
    arrow(ax, (0.57, 0.22), (0.613, 0.38))
    # outcomes
    box(ax, 0.79, 0.62, 0.195, 0.20, "Tiny selected", "reuse tiny output:\nno further inference", ec=CANDIDATE_COLORS["tiny"])
    box(ax, 0.79, 0.20, 0.195, 0.30, "Small / Medium / Large", "run ONE additional\nstatic engine after the probe\n(route = probe + decision\n+ switch + engine)",
        ec=CANDIDATE_COLORS["large"])
    arrow(ax, (0.745, 0.60), (0.788, 0.70))
    arrow(ax, (0.745, 0.42), (0.788, 0.36))
    ax.text(0.8875, 0.10, "static deployment runs only its engine:\ncost $S_h(\\ell)$, no probe", ha="center",
            fontsize=5.4, color=MUTED)
    save_all(fig, GEN / "fig2_pipeline_v20")
    (GEN / "_tmp_tiny.png").unlink(missing_ok=True)


if __name__ == "__main__":
    fig_forest()
    fig_rq2()
    fig_pipeline()
