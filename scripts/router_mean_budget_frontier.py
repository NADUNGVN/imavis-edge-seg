"""Mean-cost (soft-budget) view of routing on the deployment-matched caches.

Under a per-frame hard budget, no policy can exceed the largest feasible static
candidate except where a smaller candidate is better on individual images (oracle
headroom). The classic adaptive-inference question is different: at a given MEAN
cost, does routing beat the best mix of static candidates? This script compares the
held-out (mean cost, mIoU) points of candidate-specific routing without a per-frame
cap (policy C, `_select_by_candidate_specific_risk`) and of the oracle against the
exact random mixture of the two adjacent static candidates (randomly mixing two
adjacent static candidates realizes any point on that segment in expectation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from router_review_analyses import TAG, BASE, LEVELS, RUNS, Split, load_costs, miou, ordered_levels, scored


def c_choice(per_level: dict[str, float], target: float, order: list[str]) -> str:
    for lv in order:
        if per_level[lv] <= target:
            return lv
    return order[-1]


def static_curve(sp: Split, cost: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    order = ordered_levels(cost)
    xs = np.array([cost[lv] for lv in order])
    ys = np.array([miou(sp.test_cm[lv].sum(0)) for lv in order])
    return xs, ys


def static_mix_miou(sp: Split, cost: dict[str, float], m: float) -> float:
    """Exact expected mIoU of randomly mixing the two adjacent static candidates whose
    mean cost brackets m (expected confusion matrix of the mixture)."""
    order = ordered_levels(cost)
    for lo, hi in zip(order, order[1:]):
        if cost[lo] <= m <= cost[hi]:
            p = (m - cost[lo]) / (cost[hi] - cost[lo])
            return miou((1 - p) * sp.test_cm[lo].sum(0) + p * sp.test_cm[hi].sum(0))
    raise ValueError(m)


def main() -> None:
    costs = load_costs("median")
    rows = []
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        for split_name, raw in dump.items():
            if split_name.startswith("_"):
                continue
            sp = Split(raw)
            # dense target grid: quantiles of all fit-half predicted errors
            preds = np.concatenate([[sp.cal[lv].predict(s) for s in sp.fit_scores] for lv in LEVELS])
            targets = sorted(set(np.quantile(preds, np.linspace(0.02, 0.98, 25)).tolist()))
            for backend, cost in costs.items():
                order = ordered_levels(cost)
                xs, ys = static_curve(sp, cost)
                for t in targets:
                    choice = [c_choice({lv: sp.cal[lv].predict(s) for lv in LEVELS}, t, order) for s in sp.test_scores]
                    cms, lat = scored(choice, sp.test_cm, cost)
                    m = lat.mean()
                    if m <= xs[0] or m >= xs[-1]:
                        continue
                    rows.append({"run": run, "backend": backend, "split": split_name,
                                 "gain_points": (miou(cms.sum(0)) - static_mix_miou(sp, cost, m)) * 100,
                                 "mean_cost_frac": (m - xs[0]) / (xs[-1] - xs[0])})
                # oracle at mean cost: per image cheapest level achieving min error
                choice = [min(order, key=lambda lv: (sp.test_err[lv][i], cost[lv])) for i in range(len(sp.test_scores))]
                cms, lat = scored(choice, sp.test_cm, cost)
                rows.append({"run": run, "backend": backend, "split": split_name, "oracle": True,
                             "gain_points": (miou(cms.sum(0)) - static_mix_miou(sp, cost, float(lat.mean()))) * 100})
    c = np.array([r["gain_points"] for r in rows if not r.get("oracle")])
    by_split = {}
    for s in sorted({r["split"] for r in rows}):
        v = [r["gain_points"] for r in rows if r["split"] == s and not r.get("oracle")]
        by_split[s] = float(np.mean(v))
    out = {
        "definition": "mIoU gain (points) over the static-interpolation curve at equal held-out mean route cost",
        "routing_points": int(len(c)),
        "routing_mean_gain_points": float(c.mean()),
        "routing_share_above_static_curve": float((c > 0).mean()),
        "routing_gain_by_split": by_split,
        "per_cell_mean_gain_points": {
            f"{r}|{b}|{sp}": float(np.mean([x["gain_points"] for x in rows
                                          if x["run"] == r and x["backend"] == b and x["split"] == sp and not x.get("oracle")]))
            for r in RUNS for b in costs for sp in sorted({x["split"] for x in rows})
        },
    }
    Path(f"reports/router_mean_budget_frontier_{TAG}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
