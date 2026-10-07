"""V20 figures for RQ3b (landscape rerun, tag 20261004).

fig11_route_vs_static_v20: per-capacity direct static latency S_h and adaptive route
  latency C_h (tiny probe + entropy + decision + selected engine), same harness and
  timer boundary; the hatched segment is the routing overhead C_h - S_h.
fig12_breakeven_v20: mean-cost gain of candidate-specific routing over static mixing
  as the measured overhead is scaled by alpha in [0, 1]; the dot marks the measured
  overhead and the crossing marks the break-even overhead.

Inputs: reports/router_breakeven_20261004.json (E3 TensorRT, logits returned;
E1 Hailo-8, explicit activation, FLOAT32 vstreams).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pace_style_v13 import (CANDIDATE_COLORS, CANDIDATES, FULL_WIDTH_MM, INK, MUTED, SINGLE_WIDTH_MM,
                            apply_style, mm_to_inches, save_all)

DEVICES = {"E3": "E3 AGX Xavier (TensorRT FP16)", "E1": "E1 Hailo-8 (INT8)"}


def breakeven(alpha: list[float], overhead: list[float], gain: list[float]) -> float | None:
    for i in range(1, len(alpha)):
        if gain[i - 1] >= 0 > gain[i]:
            t = gain[i - 1] / (gain[i - 1] - gain[i])
            return overhead[i - 1] + t * (overhead[i] - overhead[i - 1])
    return None


def fig_route_vs_static(data: dict, stem: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_inches(FULL_WIDTH_MM), mm_to_inches(62)))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.86, wspace=0.22)
    x = np.arange(len(CANDIDATES))
    for ax, dev in zip(axes, DEVICES):
        s = np.array([data[dev]["static_ms"][lv] for lv in CANDIDATES])
        r = np.array([data[dev]["route_ms"][lv] for lv in CANDIDATES])
        cols = [CANDIDATE_COLORS[lv] for lv in CANDIDATES]
        ax.bar(x - 0.19, s, 0.36, color=cols, edgecolor=INK, linewidth=0.5, label="static (direct)")
        ax.bar(x + 0.19, s, 0.36, color=cols, edgecolor=INK, linewidth=0.5, alpha=0.45)
        ax.bar(x + 0.19, r - s, 0.36, bottom=s, color="white", edgecolor=INK, linewidth=0.5, hatch="////",
               label="routing overhead")
        for i in range(len(x)):
            ax.text(x[i] - 0.19, s[i], f"{s[i]:.1f}", ha="center", va="bottom", fontsize=6, color=INK)
            ax.text(x[i] + 0.19, r[i], f"{r[i]:.1f}", ha="center", va="bottom", fontsize=6, color=INK)
        ax.set_xticks(x, [lv.title() for lv in CANDIDATES])
        ax.set_ylabel("Median latency (ms)")
        ax.set_title(DEVICES[dev], fontsize=7.5)
        ax.set_ylim(0, r.max() * 1.15)
        ax.text(0.02, 0.97, f"overhead {np.min(r - s):.1f}–{np.max(r - s):.1f} ms", transform=ax.transAxes,
                fontsize=6.6, color=MUTED, va="top")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, 1.22), ncol=2, frameon=False, fontsize=6.6,
                   handles=[plt.Rectangle((0, 0), 1, 1, fc="#BBBBBB", ec=INK, lw=0.5),
                            plt.Rectangle((0, 0), 1, 1, fc="white", ec=INK, lw=0.5, hatch="////")],
                   labels=["static deployment S (left) / route through tiny C (right)", "routing overhead C − S"])
    save_all(fig, stem)


def fig_breakeven(data: dict, stem: Path) -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_inches(FULL_WIDTH_MM), mm_to_inches(58)))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.82, wspace=0.25)
    out = {}
    for ax, dev in zip(axes, DEVICES):
        ba = data[dev]["by_alpha"]
        alpha = sorted(ba, key=float)
        ov = [ba[a]["mean_overhead_ms"] for a in alpha]
        mix = [ba[a]["mixture_gain_points"] for a in alpha]
        hard = [ba[a]["D_vs_static"]["mean_delta_points"] for a in alpha]
        be = breakeven([float(a) for a in alpha], ov, mix)
        out[dev] = {"breakeven_ms": be, "measured_ms": ov[-1]}
        ax.axhline(0, color=INK, lw=0.7)
        ax.plot(ov, mix, "-o", ms=3, color="#1F6FB2", label="mean-cost budget (vs static mix)")
        ax.plot(ov, hard, "-s", ms=3, color="#D55E00", label="per-frame budget (vs best static)")
        ax.plot(ov[-1], mix[-1], "o", ms=6, mfc="none", mec=INK)
        ax.annotate(f"measured\n{ov[-1]:.1f} ms", (ov[-1], mix[-1]), xytext=(-6, 10), textcoords="offset points",
                    ha="right", fontsize=6.4, color=INK)
        if be is not None:
            ax.axvline(be, color=MUTED, lw=0.7, ls=(0, (2, 2)))
            ax.text(be, ax.get_ylim()[1], f" break-even {be:.1f} ms", fontsize=6.4, color=MUTED, va="top")
        ax.set_xlabel("Routing overhead per frame (ms)")
        ax.set_ylabel("Routing − static (mIoU points)")
        ax.set_title(DEVICES[dev], fontsize=7.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.12, 1.25), ncol=2, frameon=False, fontsize=6.6)
    save_all(fig, stem)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    a = p.parse_args()
    data = json.loads((a.repo_root / "reports/router_breakeven_20261004.json").read_text())
    gen = Path(__file__).parents[1] / "generated"
    fig_route_vs_static(data, gen / "fig11_route_vs_static_v20")
    print(json.dumps(fig_breakeven(data, gen / "fig12_breakeven_v20"), indent=1))


if __name__ == "__main__":
    main()
