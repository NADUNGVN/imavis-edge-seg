"""V19 figures added in response to review (2026-10-03).

fig10_mean_budget_gain: mIoU gain of candidate-specific routing over the exact
    random mixture of adjacent static candidates at equal held-out mean route cost,
    per condition (bars = mean; dots = run x backend means).
figS5_calibration_reliability: pooled held-out reliability of the four
    candidate-specific calibrators (equal-count bins of predicted error).
Inputs: reports/router_mean_budget_frontier_20261003.json,
        reports/router_review_analyses_20261003.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pace_style_v13 import CANDIDATE_COLORS, INK, MUTED, SINGLE_WIDTH_MM, apply_style, mm_to_inches, save_all

REPO = Path(__file__).parents[3]
OUT = Path(__file__).parents[1] / "generated"
SPLITS = [("cityscapes", "Cityscapes"), ("acdc/fog", "Fog"), ("acdc/night", "Night"), ("acdc/rain", "Rain"), ("acdc/snow", "Snow")]
BACKEND_MARK = {"E3": "o", "E1": "s"}


def mean_budget_gain() -> None:
    data = json.loads((REPO / "reports/router_mean_budget_frontier_20261003.json").read_text())
    cells = data["per_cell_mean_gain_points"]
    apply_style()
    fig, ax = plt.subplots(figsize=(mm_to_inches(SINGLE_WIDTH_MM), mm_to_inches(58)))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.93)
    for i, (key, label) in enumerate(SPLITS):
        vals = {k: v for k, v in cells.items() if k.split("|")[2] == key}
        mean = float(np.mean(list(vals.values())))
        ax.bar(i, mean, width=0.62, color="#9CC5E3", edgecolor=CANDIDATE_COLORS["large"], linewidth=0.8, zorder=2)
        for j, (k, v) in enumerate(sorted(vals.items())):
            backend = k.split("|")[1]
            ax.scatter(i - 0.18 + 0.072 * j, v, s=11, marker=BACKEND_MARK[backend], color=INK, zorder=3, linewidths=0)
        ax.text(i, max(vals.values()) + 0.15, f"+{mean:.2f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(SPLITS)), [label for _, label in SPLITS])
    ax.set_ylabel("Gain over static mix (mIoU pts)")
    ax.set_ylim(min(0, ax.get_ylim()[0]), ax.get_ylim()[1] * 1.22)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    handles = [plt.Line2D([], [], marker=m, linestyle="none", color=INK, markersize=3.5, label=f"{b} run means")
               for b, m in BACKEND_MARK.items()]
    ax.legend(handles=handles, frameon=False, fontsize=6.4, loc="upper right", ncol=2, handletextpad=0.2)
    save_all(fig, OUT / "fig10_mean_budget_gain_v19")
    plt.close(fig)


def reliability(tag: str = "20261003", version: str = "v19") -> None:
    data = json.loads((REPO / f"reports/router_review_analyses_{tag}.json").read_text())["reliability"]
    apply_style()
    fig, ax = plt.subplots(figsize=(mm_to_inches(SINGLE_WIDTH_MM), mm_to_inches(70)))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.15, top=0.97)
    lim = 0.33
    ax.plot([0, lim], [0, lim], color=MUTED, linestyle="--", linewidth=0.8, label="perfect calibration")
    for level, pts in data.items():
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], marker="o", markersize=3.2, linewidth=1.2, color=CANDIDATE_COLORS[level],
                label=level.title())
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted per-image pixel error")
    ax.set_ylabel("Observed per-image pixel error")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")
    save_all(fig, OUT / f"figS5_calibration_reliability_{version}")
    plt.close(fig)


if __name__ == "__main__":
    mean_budget_gain()
    reliability()
    print("ok")
