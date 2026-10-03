"""Reviewer-requested router analyses on the deployment-matched caches (2026-10-03).

Pure post-processing of `reports/router_deployment_matched_20260929/` (per-image
dumps + evaluation JSONs) and the measured overhead reports -- no re-inference.
Every operating point is still selected on fit-half statistics only, exactly as in
`scripts/replay_router_with_overhead.py::replay_e2e_aware`, before held-out labels
are read.

Analyses
1. Policy decomposition. A (rank escalation, no budget), A-hard (A's choice clipped
   to the most expensive in-budget route: hard budget, *no* candidate-specific risk),
   D, pooled D, best static feasible candidate, and the per-image GT oracle.
2. Cost-statistic sensitivity. Cost table AND budget grid built from the median,
   p95, or p99 complete-route latency.
3. Image-level bootstrap CIs for macro delta-mIoU. One resample of held-out images
   per split per replicate, shared by every run/backend/budget cell of that split.
4. Calibration quality on held-out halves (MAE/RMSE/Spearman per candidate),
   bin-count sensitivity (5/10/20 equal-count bins, isotonic/PAVA), and the
   pixel-error vs per-image (1 - mIoU) agreement of the risk target.
5. Leave-one-condition-out (LOCO): calibrators, risk-target grid, and operating
   points fitted on the other four conditions' fit halves only; evaluated on the
   held-out half of the unseen condition.
"""

from __future__ import annotations

import argparse
import os
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from imavis_edge_seg.router.calibrator import RiskCalibrator, fit_risk_calibrator
from imavis_edge_seg.router.grid import macro_quantile_grid, select_budget_matched_operating_point

LEVELS = ("tiny", "small", "medium", "large")
RUNS = ("a", "b", "c")
BACKENDS = {"E3": ("reports/router_overhead_E3_20260922.json", "gpu"), "E1": ("reports/router_overhead_E1_20260928.json", "numpy")}
STATS = {"median": "median_ms", "p95": "p95_ms", "p99": "p99_ms"}
# Re-targetable for the landscape rerun (2026-10-04): PACE_ROUTER_BASE points at the new
# evaluate_router dumps, PACE_TAG is the date suffix of measurement/result files, and
# PACE_ROUTE_COSTS=same_harness takes route costs from reports/static_vs_route_*_<TAG>.json.
BASE = Path(os.environ.get("PACE_ROUTER_BASE", "reports/router_deployment_matched_20260929"))
TAG = os.environ.get("PACE_TAG", "20261003")


# --------------------------------------------------------------------------- data
class Split:
    def __init__(self, dump: dict[str, Any]) -> None:
        self.fit_scores = np.asarray(dump["fit_raw_scores"], dtype=np.float64)
        self.test_scores = np.asarray(dump["test_raw_scores"], dtype=np.float64)
        self.fit_cm = {lv: np.asarray(dump["fit_confusion_matrices"][lv], dtype=np.int64) for lv in LEVELS}
        self.test_cm = {lv: np.asarray(dump["test_confusion_matrices"][lv], dtype=np.int64) for lv in LEVELS}
        self.cal = {lv: RiskCalibrator.from_dict(d) for lv, d in dump["per_level_calibrators"].items()}
        self.pooled = {lv: RiskCalibrator.from_dict(d) for lv, d in dump["pooled_per_level_calibrators"].items()}
        self.probe = dump["probe_level"]
        self.fit_err = {lv: pixel_error(self.fit_cm[lv]) for lv in LEVELS}
        self.test_err = {lv: pixel_error(self.test_cm[lv]) for lv in LEVELS}


def pixel_error(cms: np.ndarray) -> np.ndarray:
    tot = cms.sum(axis=(1, 2))
    tr = np.trace(cms, axis1=1, axis2=2)
    return np.where(tot > 0, 1.0 - tr / np.maximum(tot, 1), 0.0)


def image_miou_error(cms: np.ndarray) -> np.ndarray:
    tp = np.diagonal(cms, axis1=1, axis2=2).astype(np.float64)
    union = cms.sum(axis=1) + cms.sum(axis=2) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    return 1.0 - np.nanmean(iou, axis=1)


def miou(total: np.ndarray) -> float:
    tp = np.diagonal(total).astype(np.float64)
    union = total.sum(axis=0) + total.sum(axis=1) - tp
    iou = np.full_like(tp, np.nan)
    np.divide(tp, union, out=iou, where=union > 0)
    return float(np.nanmean(iou))


def load_costs(stat: str) -> dict[str, dict[str, float]]:
    if os.environ.get("PACE_ROUTE_COSTS") == "same_harness":
        key = {"median": "median_ms", "p95": "p95_ms", "p99": "p99_ms"}[stat]
        files = {"E3": (f"reports/static_vs_route_E3_{TAG}.json", "|logits"),
                 "E1": (f"reports/static_vs_route_E1_explicit_float32_{TAG}.json", "")}
        out = {}
        for backend, (path, suffix) in files.items():
            res = json.loads(Path(path).read_text())["results"]
            out[backend] = {lv: float(res[f"route_tiny->{lv}{suffix}"][key]) for lv in LEVELS}
        return out
    out = {}
    for backend, (path, entropy) in BACKENDS.items():
        warm = json.loads(Path(path).read_text())["warm"][entropy]
        out[backend] = {lv: float(warm[f"tiny->{lv}"][STATS[stat]]) for lv in LEVELS}
    return out


# ------------------------------------------------------------------------ policies
def ordered_levels(cost: dict[str, float]) -> list[str]:
    return sorted(LEVELS, key=lambda lv: cost[lv])


def rank_choice(risk: float, target: float, order: list[str]) -> str:
    """policy._select_by_risk"""
    if risk <= target:
        return order[0]
    return order[min(int(risk / target), len(order) - 1)]


def d_choice(per_level: dict[str, float], target: float, budget: float, cost: dict[str, float], order: list[str]) -> str:
    """policy._select_by_risk_and_latency_budget"""
    inb = [lv for lv in order if cost[lv] <= budget]
    if not inb:
        return order[0]
    meet = [lv for lv in inb if per_level[lv] <= target]
    if meet:
        return min(meet, key=lambda lv: cost[lv])
    return min(inb, key=lambda lv: (per_level[lv], cost[lv]))


def decide(policy: str, scores: np.ndarray, target: float, budget: float | None,
           cost: dict[str, float], cal: dict[str, RiskCalibrator], probe: str) -> list[str]:
    order = ordered_levels(cost)
    out = []
    for s in scores:
        if policy == "A":
            out.append(rank_choice(cal[probe].predict(s), target, order))
        elif policy == "A_hard":
            choice = rank_choice(cal[probe].predict(s), target, order)
            if cost[choice] > budget:
                inb = [lv for lv in order if cost[lv] <= budget] or [order[0]]
                choice = inb[-1]
            out.append(choice)
        elif policy == "D":
            out.append(d_choice({lv: c.predict(s) for lv, c in cal.items()}, target, budget, cost, order))
        else:
            raise ValueError(policy)
    return out


def scored(choice: list[str], cms: dict[str, np.ndarray], cost: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Per-image confusion matrices and per-image route cost of a decision vector."""
    idx = {lv: i for i, lv in enumerate(LEVELS)}
    stack = np.stack([cms[lv] for lv in LEVELS])  # (L, n, C, C)
    sel = np.array([idx[c] for c in choice])
    per_image = stack[sel, np.arange(len(choice))]
    lat = np.array([cost[c] for c in choice])
    return per_image, lat


def run_policy(policy: str, sp: Split, targets: list[float], budgets: list[float], cost: dict[str, float],
               cal: dict[str, RiskCalibrator], fit_scores=None, fit_cm=None, test_scores=None, test_cm=None) -> dict[float, dict]:
    """Fit-half operating-point selection identical to replay_e2e_aware."""
    fit_scores = sp.fit_scores if fit_scores is None else fit_scores
    fit_cm = sp.fit_cm if fit_cm is None else fit_cm
    test_scores = sp.test_scores if test_scores is None else test_scores
    test_cm = sp.test_cm if test_cm is None else test_cm
    res = {}
    budget_free = policy == "A"
    frontier = None
    for b in budgets:
        if budget_free and frontier is not None:
            points = frontier
        else:
            points = []
            for t in targets:
                bd = None if budget_free else b
                fc, fl = scored(decide(policy, fit_scores, t, bd, cost, cal, sp.probe), fit_cm, cost)
                points.append((float(fl.mean()), miou(fc.sum(0)), t))
            if budget_free:
                frontier = points
        t, feas = select_budget_matched_operating_point(points, b)
        choice = decide(policy, test_scores, t, None if budget_free else b, cost, cal, sp.probe)
        cms, lat = scored(choice, test_cm, cost)
        res[b] = {"target": t, "per_image_cm": cms, "lat": lat, "miou": miou(cms.sum(0)),
                  "violation": float(np.mean(lat > b + 1e-9)), "choice": choice}
    return res


def static_best(sp: Split, budgets: list[float], cost: dict[str, float], test_cm=None) -> dict[float, dict]:
    test_cm = sp.test_cm if test_cm is None else test_cm
    order = ordered_levels(cost)
    res = {}
    for b in budgets:
        lv = ([x for x in order if cost[x] <= b] or [order[0]])[-1]
        cms = test_cm[lv]
        res[b] = {"per_image_cm": cms, "miou": miou(cms.sum(0)), "violation": float(cost[lv] > b + 1e-9), "level": lv,
                  "lat": np.full(len(cms), cost[lv])}
    return res


def oracle(sp: Split, budgets: list[float], cost: dict[str, float]) -> dict[float, dict]:
    order = ordered_levels(cost)
    res = {}
    for b in budgets:
        inb = [x for x in order if cost[x] <= b] or [order[0]]
        choice = [min(inb, key=lambda lv: sp.test_err[lv][i]) for i in range(len(sp.test_scores))]
        cms, lat = scored(choice, sp.test_cm, cost)
        res[b] = {"per_image_cm": cms, "miou": miou(cms.sum(0)), "lat": lat}
    return res


# ----------------------------------------------------------------- cell analysis
def build_cells(splits: dict[str, dict[str, Split]], grids: dict[str, list[float]], stat: str) -> list[dict]:
    costs = load_costs(stat)
    cells = []
    for run in RUNS:
        for backend, cost in costs.items():
            budgets = sorted(set(cost.values()))
            for split_name, sp in splits[run].items():
                targets = grids[run][split_name]
                pol = {
                    "A": run_policy("A", sp, targets, budgets, cost, sp.cal),
                    "A_hard": run_policy("A_hard", sp, targets, budgets, cost, sp.cal),
                    "D": run_policy("D", sp, targets, budgets, cost, sp.cal),
                    "D_pooled": run_policy("D", sp, targets, budgets, cost, sp.pooled),
                    "static": static_best(sp, budgets, cost),
                    "oracle": oracle(sp, budgets, cost),
                }
                for b in budgets:
                    cells.append({"run": run, "backend": backend, "split": split_name, "budget": b,
                                  **{k: v[b] for k, v in pol.items()}})
    return cells


def compare(cells: list[dict], x: str, y: str, fair_on: str | None) -> dict:
    rows = [c for c in cells if fair_on is None or c[fair_on]["violation"] == 0.0]
    d = np.array([c[x]["miou"] - c[y]["miou"] for c in rows])
    return {"cells": len(rows), "wins": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
            "losses": int((d < -1e-12).sum()), "mean_delta_points": float(d.mean() * 100) if len(d) else None}


def bootstrap(cells: list[dict], x: str, y: str, fair_on: str | None, reps: int, seed: int = 0) -> dict:
    rows = [c for c in cells if fair_on is None or c[fair_on]["violation"] == 0.0]
    rng = np.random.default_rng(seed)
    sizes = {c["split"]: len(c[x]["per_image_cm"]) for c in rows}
    vals = []
    for _ in range(reps):
        idx = {s: rng.integers(0, n, n) for s, n in sizes.items()}
        deltas = [miou(c[x]["per_image_cm"][idx[c["split"]]].sum(0)) - miou(c[y]["per_image_cm"][idx[c["split"]]].sum(0))
                  for c in rows]
        vals.append(np.mean(deltas) * 100)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"ci95_points": [float(lo), float(hi)], "reps": reps}


def violation_counts(cells: list[dict]) -> dict:
    return {k: int(sum(c[k]["violation"] > 0 for c in cells)) for k in ("A", "A_hard", "D", "D_pooled", "static")}


def mean_latency_ratio(cells: list[dict], x: str, y: str) -> dict:
    """Held-out mean route cost of x relative to y, per cell, then averaged."""
    r = np.array([c[x]["lat"].mean() / c[y]["lat"].mean() for c in cells])
    saved = np.array([1 - c[x]["lat"].mean() / c[y]["lat"].mean() for c in cells])
    return {"mean_cost_ratio": float(r.mean()), "mean_cost_saving_pct": float(saved.mean() * 100),
            "cells_with_saving": int((saved > 1e-9).sum())}


def mean_miou(cells: list[dict]) -> dict:
    return {k: float(np.mean([c[k]["miou"] for c in cells]) * 100) for k in ("A", "A_hard", "D", "D_pooled", "static", "oracle")}


# --------------------------------------------------------------- calibration
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pava(x: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float)
    blocks = [[v, 1.0, xs[i], xs[i]] for i, v in enumerate(ys)]  # mean, weight, xmin, xmax
    merged: list[list[float]] = []
    for blk in blocks:
        merged.append(blk)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            b2 = merged.pop()
            b1 = merged.pop()
            w = b1[1] + b2[1]
            merged.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w, w, b1[2], b2[3]])
    edges = np.array([m[2] for m in merged])
    means = np.array([m[0] for m in merged])

    def predict(s: float) -> float:
        i = int(np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(means) - 1))
        return float(means[i])
    return predict


def calibration_quality(splits: dict[str, dict[str, Split]]) -> dict:
    out: dict[str, Any] = {"configured_heldout": {}, "bin_sensitivity_mae": {}, "risk_target_agreement": {}}
    for split_name in splits["a"]:
        per_level = {}
        for lv in LEVELS:
            mae, rmse, rho = [], [], []
            for run in RUNS:
                sp = splits[run][split_name]
                pred = np.array([sp.cal[lv].predict(s) for s in sp.test_scores])
                obs = sp.test_err[lv]
                mae.append(np.mean(np.abs(pred - obs)))
                rmse.append(np.sqrt(np.mean((pred - obs) ** 2)))
                rho.append(spearman(sp.test_scores, obs))
            per_level[lv] = {"mae": float(np.mean(mae)), "rmse": float(np.mean(rmse)), "spearman_score_vs_error": float(np.mean(rho)),
                             "mean_observed_error": float(np.mean([splits[r][split_name].test_err[lv].mean() for r in RUNS]))}
        out["configured_heldout"][split_name] = per_level
        sens = {}
        for name in ("bins5", "bins10", "bins20", "isotonic"):
            maes = []
            for run in RUNS:
                sp = splits[run][split_name]
                for lv in LEVELS:
                    if name == "isotonic":
                        f = pava(sp.fit_scores, sp.fit_err[lv])
                    else:
                        cal = fit_risk_calibrator(sp.fit_scores, sp.fit_err[lv], num_bins=int(name[4:]))
                        f = cal.predict
                    pred = np.array([f(s) for s in sp.test_scores])
                    maes.append(np.mean(np.abs(pred - sp.test_err[lv])))
            sens[name] = float(np.mean(maes))
        out["bin_sensitivity_mae"][split_name] = sens
        rho = []
        for run in RUNS:
            sp = splits[run][split_name]
            for lv in LEVELS:
                rho.append(spearman(sp.test_err[lv], image_miou_error(sp.test_cm[lv])))
        out["risk_target_agreement"][split_name] = {"spearman_pixel_error_vs_image_1_minus_miou": float(np.mean(rho))}
    return out


def reliability_points(splits: dict[str, dict[str, Split]], bins: int = 10) -> dict:
    """Pooled held-out reliability per candidate: mean predicted vs mean observed in
    equal-count bins of predicted error, pooled over runs and splits."""
    out = {}
    for lv in LEVELS:
        pred, obs = [], []
        for run in RUNS:
            for sp in splits[run].values():
                pred.extend(sp.cal[lv].predict(s) for s in sp.test_scores)
                obs.extend(sp.test_err[lv])
        pred_a, obs_a = np.array(pred), np.array(obs)
        order = np.argsort(pred_a)
        chunks = np.array_split(order, bins)
        out[lv] = [[float(pred_a[c].mean()), float(obs_a[c].mean())] for c in chunks]
    return out


# ---------------------------------------------------------------------- LOCO
def loco(splits: dict[str, dict[str, Split]], stat: str) -> dict:
    costs = load_costs(stat)
    names = list(splits["a"])
    result: dict[str, Any] = {}
    rows_vs_a, rows_vs_cfg = [], []
    for held in names:
        others = [n for n in names if n != held]
        for run in RUNS:
            S = splits[run]
            pooled_scores = np.concatenate([S[n].fit_scores for n in others])
            pooled_err = {lv: np.concatenate([S[n].fit_err[lv] for n in others]) for lv in LEVELS}
            cal = {lv: fit_risk_calibrator(pooled_scores, pooled_err[lv]) for lv in LEVELS}
            grid = macro_quantile_grid([np.concatenate([S[n].fit_err[lv] for lv in LEVELS]) for n in others])
            fit_cm = {lv: np.concatenate([S[n].fit_cm[lv] for n in others]) for lv in LEVELS}
            sp = S[held]
            for backend, cost in costs.items():
                budgets = sorted(set(cost.values()))
                kw = dict(fit_scores=pooled_scores, fit_cm=fit_cm)
                d = run_policy("D", sp, grid, budgets, cost, cal, **kw)
                a = run_policy("A", sp, grid, budgets, cost, cal, **kw)
                ah = run_policy("A_hard", sp, grid, budgets, cost, cal, **kw)
                for b in budgets:
                    rows_vs_a.append({"held": held, "d": d[b], "a": a[b], "ah": ah[b]})
    for key, other, fair in (("vs_A_fair", "a", True), ("vs_A_hard", "ah", False)):
        rows = [r for r in rows_vs_a if not fair or r["a"]["violation"] == 0.0]
        dl = np.array([r["d"]["miou"] - r[other]["miou"] for r in rows])
        result[key] = {"cells": len(rows), "wins": int((dl > 1e-12).sum()), "ties": int((np.abs(dl) <= 1e-12).sum()),
                       "losses": int((dl < -1e-12).sum()), "mean_delta_points": float(dl.mean() * 100)}
    result["D_violating_cells"] = int(sum(r["d"]["violation"] > 0 for r in rows_vs_a))
    result["total_cells"] = len(rows_vs_a)
    result["per_condition_mean_D_miou"] = {
        h: float(np.mean([r["d"]["miou"] for r in rows_vs_a if r["held"] == h]) * 100) for h in names}
    return result


# ---------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--output-json", type=Path, default=Path(f"reports/router_review_analyses_{TAG}.json"))
    args = p.parse_args()

    splits: dict[str, dict[str, Split]] = {}
    grids: dict[str, dict[str, list[float]]] = {}
    for run in RUNS:
        dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
        ev = json.loads((BASE / f"run_{run}_evaluation.json").read_text())
        splits[run] = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
        grids[run] = {k: ev[k]["risk_target_grid"] for k in splits[run]}

    out: dict[str, Any] = {"_definition": {
        "A_hard": "policy A's rank-escalation choice; if its measured route cost exceeds the budget, replaced by the most expensive in-budget route",
        "static": "largest static candidate whose measured route cost fits the budget",
        "fair_cell": "policy A has zero held-out budget violations",
        "delta_units": "mIoU percentage points",
        "bootstrap": "held-out images resampled per split, shared across all cells of that split",
    }}
    for stat in STATS:
        print(f"[{stat}] building cells")
        cells = build_cells(splits, grids, stat)
        block = {
            "cells": len(cells),
            "violating_cells": violation_counts(cells),
            "mean_miou_points": mean_miou(cells),
            "D_vs_A_fair": compare(cells, "D", "A", "A"),
            "D_pooled_vs_A_fair": compare(cells, "D_pooled", "A", "A"),
            "D_vs_A_hard": compare(cells, "D", "A_hard", None),
            "A_hard_vs_A_fair": compare(cells, "A_hard", "A", "A"),
            "D_vs_static": compare(cells, "D", "static", None),
            "D_cost_vs_static": mean_latency_ratio(cells, "D", "static"),
            "D_cost_vs_A_hard": mean_latency_ratio(cells, "D", "A_hard"),
            "oracle_minus_D_points": float(np.mean([c["oracle"]["miou"] - c["D"]["miou"] for c in cells]) * 100),
        }
        if stat in ("median", "p95"):
            print(f"[{stat}] bootstrapping")
            block["D_vs_A_fair"].update(bootstrap(cells, "D", "A", "A", args.bootstrap_reps))
            block["D_vs_A_hard"].update(bootstrap(cells, "D", "A_hard", None, args.bootstrap_reps))
            block["D_vs_static"].update(bootstrap(cells, "D", "static", None, args.bootstrap_reps))
        out[stat] = block
        print(json.dumps({k: v for k, v in block.items()}, indent=1))
    print("[calibration]")
    out["calibration"] = calibration_quality(splits)
    out["reliability"] = reliability_points(splits)
    print("[loco]")
    out["loco_median"] = loco(splits, "median")
    out["loco_p95"] = loco(splits, "p95")
    args.output_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["loco_median"], indent=1))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
