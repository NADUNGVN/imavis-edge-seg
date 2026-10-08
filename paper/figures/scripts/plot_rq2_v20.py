"""V20 Fig. 6 (RQ2): absolute mIoU of the elastic large capacity, the same architecture
trained alone (PACE-Large), and Fast-SCNN, per split, mean +- std over seeds 0-2.
Elastic tiny/small/medium are drawn as short ticks on the elastic bar for context.

Inputs: reports/landscape_20261004/eval_{supernet,pace_large,fast_scnn}_seed{0,1,2}.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pace_style_v13 import (CANDIDATE_COLORS, FULL_WIDTH_MM, INK, MUTED, apply_style, mm_to_inches,
                            save_all)

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports" / "landscape_20261004"
SPLITS = [("cityscapes", "Cityscapes"), ("acdc/fog", "ACDC fog"), ("acdc/night", "ACDC night"),
          ("acdc/rain", "ACDC rain"), ("acdc/snow", "ACDC snow")]


def load(name: str, level: str | None = None) -> np.ndarray:
    """Return seeds x splits mIoU in points."""
    rows = []
    for s in range(3):
        d = json.loads((REPORTS / f"eval_{name}_seed{s}.json").read_text())
        d = d[level] if level else d
        rows.append([100 * d[k]["miou"] for k, _ in SPLITS])
    return np.array(rows)


def main() -> None:
    apply_style()
    series = [("Elastic large (shared)", load("supernet", "large"), CANDIDATE_COLORS["large"]),
              ("PACE-Large (trained alone)", load("pace_large"), "#9CA3AF"),
              ("Fast-SCNN", load("fast_scnn"), "#E5E7EB")]
    fig, ax = plt.subplots(figsize=(mm_to_inches(FULL_WIDTH_MM), mm_to_inches(58)))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.84)
    x = np.arange(len(SPLITS))
    w = 0.26
    for i, (label, a, color) in enumerate(series):
        m, sd = a.mean(0), a.std(0, ddof=1)
        ax.bar(x + (i - 1) * w, m, w, yerr=sd, color=color, edgecolor=INK, linewidth=0.5,
               capsize=2, error_kw={"linewidth": 0.6}, label=label)
        for xi, mi in zip(x, m):
            ax.text(xi + (i - 1) * w, mi + 1.6, f"{mi:.1f}", ha="center", va="bottom", fontsize=5.5, color=INK)
    for lvl in ("tiny", "small", "medium"):
        m = load("supernet", lvl).mean(0)
        ax.hlines(m, x - 1.5 * w + 0.02, x - 0.5 * w - 0.02, colors=CANDIDATE_COLORS[lvl], linewidth=1.4)
    ax.plot([], [], color=MUTED, linewidth=1.4, label="Elastic tiny / small / medium")
    ax.set_xticks(x, [n for _, n in SPLITS])
    ax.set_ylabel("mIoU (%)")
    ax.set_ylim(0, 72)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False)
    save_all(fig, ROOT / "paper" / "figures" / "fig6_rq2_shared_vs_alone_v20")


if __name__ == "__main__":
    main()
