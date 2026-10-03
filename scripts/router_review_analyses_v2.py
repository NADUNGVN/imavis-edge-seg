"""Second-round review analyses (2026-10-03), on the same deployment-matched caches.

1. Fair static cost. Static deployment of one candidate needs no tiny probe,
   entropy, calibrator, or policy, so it is charged its direct static latency
   S_h(l) from the candidate-only benchmark LUT (TensorRT trtexec on E3; hailortcli
   with a continuously activated network group on E1), not the adaptive route
   cost C_h(l). Routing policies keep C_h. Budgets are the union of both tables'
   values per device.
2. Scalar-threshold-hard baseline (T-hard): same probe score s(x), same hard
   feasibility mask, but three monotone thresholds on s(x) map the score to
   tiny/small/medium/large; thresholds are chosen on the fit half to maximize
   fit-half mIoU subject to fit-half mean route cost <= budget, then the chosen
   level is clipped to the most expensive feasible route.
3. Routing selection frequencies of D per backend/condition/budget.
4. Mean-cost comparison against static mixtures under S_h (mIoU of the expected
   confusion matrix of the mixture).
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from router_review_analyses import (
    BASE,
    LEVELS,
    RUNS,
    Split,
    decide,
    load_costs,
    miou,
    ordered_levels,
    run_policy,
    scored,
)
from imavis_edge_seg.router.grid import select_budget_matched_operating_point

LUT_FIELD = {"median": "end_to_end_p50_ms", "p95": "end_to_end_p95_ms", "p99": "end_to_end_p99_ms"}
LUT_BACKEND = {"E1": "hailo_hef", "E3": "tensorrt_gpu"}


def load_static_costs(stat: str) -> dict[str, dict[str, float]]:
    rows = [r for r in csv.DictReader(open("outputs/benchmark_lookup_table.csv")) if r["run_index"] == "-1"]
    out: dict[str, dict[str, float]] = {}
    for dev, backend in LUT_BACKEND.items():
        out[dev] = {r["level"]: float(r[LUT_FIELD[stat]]) for r in rows
                    if r["device_id"] == dev and r["backend"] == backend and r["level"] in LEVELS}
    return out


def threshold_policy(sp: Split, budgets: list[float], cost: dict[str, float],
                     fit_scores=None, fit_cm=None) -> dict[float, dict]:
    fit_scores = sp.fit_scores if fit_scores is None else fit_scores
    fit_cm = sp.fit_cm if fit_cm is None else fit_cm
    order = ordered_levels(cost)
    qs = np.unique(np.quantile(fit_scores, np.linspace(0, 1, 13)))
    cand = [(-np.inf, *t) for t in itertools.combinations_with_replacement(np.concatenate([[-np.inf], qs, [np.inf]]), 3)]

    def choose(scores: np.ndarray, th: tuple, b: float) -> list[str]:
        idx = np.searchsorted(np.array(th[1:]), scores, side="right")
        inb = [lv for lv in order if cost[lv] <= b] or [order[0]]
        out = []
        for i in idx:
            lv = order[int(i)]
            out.append(lv if cost[lv] <= b else inb[-1])
        return out

    res = {}
    for b in budgets:
        points = []
        for th in cand:
            c = choose(fit_scores, th, b)
            cms, lat = scored(c, fit_cm, cost)
            points.append((float(lat.mean()), miou(cms.sum(0)), th))
        th, _ = select_budget_matched_operating_point(points, b)
        c = choose(sp.test_scores, th, b)
        cms, lat = scored(c, sp.test_cm, cost)
        res[b] = {"per_image_cm": cms, "lat": lat, "miou": miou(cms.sum(0)),
                  "violation": float(np.mean(lat > b + 1e-9)), "choice": c}
    return res


def static_fair(sp: Split, budgets: list[float], scost: dict[str, float]) -> dict[float, dict]:
    order = ordered_levels(scost)
    res = {}
    for b in budgets:
        inb = [x for x in order if scost[x] <= b]
        if not inb:
            res[b] = None
            continue
        lv = inb[-1]
        res[b] = {"per_image_cm": sp.test_cm[lv], "miou": miou(sp.test_cm[lv].sum(0)), "level": lv,
                  "lat": np.full(len(sp.test_scores), scost[lv]), "violation": 0.0}
    return res


def compare(rows: list[dict], x: str, y: str) -> dict:
    rows = [r for r in rows if r[x] is not None and r[y] is not None]
    d = np.array([r[x]["miou"] - r[y]["miou"] for r in rows])
    cost = np.array([r[x]["lat"].mean() / r[y]["lat"].mean() for r in rows])
    return {"cells": len(rows), "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
            "losses": int((d < -1e-12).sum()), "mean_delta_points": float(d.mean() * 100),
            "mean_cost_ratio_x_over_y": float(cost.mean())}


def bootstrap(rows: list[dict], x: str, y: str, reps: int = 1000, seed: int = 0) -> list[float]:
    rows = [r for r in rows if r[x] is not None and r[y] is not None]
    rng = np.random.default_rng(seed)
    sizes = {r["split"]: len(r[x]["per_image_cm"]) for r in rows}
    vals = []
    for _ in range(reps):
        idx = {s: rng.integers(0, n, n) for s, n in sizes.items()}
        vals.append(np.mean([miou(r[x]["per_image_cm"][idx[r["split"]]].sum(0)) -
                             miou(r[y]["per_image_cm"][idx[r["split"]]].sum(0)) for r in rows]) * 100)
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def mixture_gain(splits: dict[str, dict[str, Split]], route_costs: dict, static_costs: dict) -> dict:
    """D-style candidate-specific routing without a per-frame cap (policy C) on C_h,
    vs the mIoU of the expected confusion matrix of mixing adjacent statics on S_h."""
    from router_mean_budget_frontier import c_choice
    gains, by_split, above = [], {}, []
    for run in RUNS:
        for split_name, sp in splits[run].items():
            preds = np.concatenate([[sp.cal[lv].predict(s) for s in sp.fit_scores] for lv in LEVELS])
            targets = sorted(set(np.quantile(preds, np.linspace(0.02, 0.98, 25)).tolist()))
            for dev in route_costs:
                rc, sc = route_costs[dev], static_costs[dev]
                order = ordered_levels(rc)
                so = ordered_levels(sc)
                for t in targets:
                    choice = [c_choice({lv: sp.cal[lv].predict(s) for lv in LEVELS}, t, order) for s in sp.test_scores]
                    cms, lat = scored(choice, sp.test_cm, rc)
                    m = float(lat.mean())
                    if m < sc[so[0]] or m > sc[so[-1]]:
                        if m > sc[so[-1]]:
                            ref = miou(sp.test_cm[so[-1]].sum(0))
                        else:
                            continue
                    else:
                        for lo, hi in zip(so, so[1:]):
                            if sc[lo] <= m <= sc[hi]:
                                p = (m - sc[lo]) / (sc[hi] - sc[lo])
                                ref = miou((1 - p) * sp.test_cm[lo].sum(0) + p * sp.test_cm[hi].sum(0))
                                break
                    g = (miou(cms.sum(0)) - ref) * 100
                    gains.append(g)
                    above.append(g > 0)
                    by_split.setdefault(f"{dev}|{split_name}", []).append(g)
    return {"points": len(gains), "mean_gain_points": float(np.mean(gains)), "share_above": float(np.mean(above)),
            "by_backend_split": {k: float(np.mean(v)) for k, v in by_split.items()}}


def main() -> None:
    splits, grids = {}, {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        ev = json.loads((BASE / f"run_{run}_evaluation.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
        grids[run] = {k: ev[k]["risk_target_grid"] for k in splits[run]}

    out: dict[str, Any] = {}
    for stat in ("median", "p95", "p99"):
        rcost, scost = load_costs(stat), load_static_costs(stat)
        rows, freq = [], {}
        for run in RUNS:
            for dev in rcost:
                budgets = sorted(set(rcost[dev].values()))  # same grid as the paper
                for split_name, sp in splits[run].items():
                    t = grids[run][split_name]
                    D = run_policy("D", sp, t, budgets, rcost[dev], sp.cal)
                    AH = run_policy("A_hard", sp, t, budgets, rcost[dev], sp.cal)
                    TH = threshold_policy(sp, budgets, rcost[dev])
                    ST = static_fair(sp, budgets, scost[dev])
                    for b in budgets:
                        rows.append({"run": run, "dev": dev, "split": split_name, "budget": b,
                                     "D": D[b], "A_hard": AH[b], "T_hard": TH[b], "static_S": ST[b]})
                        if stat == "median":
                            key = f"{dev}|{split_name}|{b:.2f}"
                            f = freq.setdefault(key, {lv: 0 for lv in LEVELS})
                            for c in D[b]["choice"]:
                                f[c] += 1
        block = {
            "D_vs_T_hard": compare(rows, "D", "T_hard"),
            "T_hard_vs_A_hard": compare(rows, "T_hard", "A_hard"),
            "D_vs_static_S": compare(rows, "D", "static_S"),
            "T_hard_violating_cells": int(sum(r["T_hard"]["violation"] > 0 for r in rows)),
            "static_S_level_by_dev_budget": sorted({(r["dev"], round(r["budget"], 2), r["static_S"]["level"]) for r in rows if r["static_S"]}),
        }
        if stat == "median":
            block["D_vs_T_hard"]["ci95_points"] = bootstrap(rows, "D", "T_hard")
            block["D_vs_static_S"]["ci95_points"] = bootstrap(rows, "D", "static_S")
            block["mixture_vs_static_S"] = mixture_gain(splits, rcost, scost)
            out["selection_frequency_D_median"] = freq
        out[stat] = block
        print(stat, json.dumps({k: v for k, v in block.items() if k != "static_S_level_by_dev_budget"}, indent=1))
        print(block["static_S_level_by_dev_budget"])
    Path("reports/router_review_analyses_v2_20261003.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/router_review_analyses_v2_20261003.json")


if __name__ == "__main__":
    main()
