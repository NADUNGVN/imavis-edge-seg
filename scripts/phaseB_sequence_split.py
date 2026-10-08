"""Phase B robustness: sequence-grouped fit/held-out split (replay from stored predictions).

The per-image dumps store the alternating-index halves separately (fit = even manifest
index 2i, held-out = odd index 2i+1). We reassemble each validation split in manifest
order, assign every image a sequence group, and re-split by WHOLE GROUPS:
  Cityscapes: city + sequence id (e.g. frankfurt_000001);  ACDC: recording id (e.g. GOPR0351).
Groups are assigned greedily, largest first, to whichever half is currently smaller, so
no sequence contributes frames to both halves. Everything that the main analysis fits on
the fit half is refitted on the new fit half only: per-condition and pooled calibrators,
risk-target grid, T-hard thresholds, and operating points. Held-out labels are used only
for scoring. No re-inference, no new training, no new measurements.

Caveat (reported, not hidden): some splits have very few groups (ACDC night: 2, fog: 3),
so the new held-out half of these conditions is a different recording, i.e. a harder
cross-sequence test, and the split is coarse.
Output: reports/phaseB_sequence_split_20261004.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from imavis_edge_seg.router.calibrator import fit_risk_calibrator
from imavis_edge_seg.router.grid import macro_quantile_grid
from router_review_analyses import BASE, LEVELS, RUNS, compare, load_costs, pixel_error, run_policy, static_best
from router_review_analyses_v2 import compare as compare2, static_fair, threshold_policy
from router_same_harness_analysis import configurations

SPLITS = ("cityscapes", "acdc/fog", "acdc/night", "acdc/rain", "acdc/snow")


def manifest(split):
    path = "data/manifests/cityscapes_val.csv" if split == "cityscapes" else "data/manifests/acdc_val.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if split != "cityscapes":
        rows = [r for r in rows if r["image_path"].split("/")[1] == split.split("/")[1]]
    return rows


def group_of(split, row):
    name = row["image_path"].split("/")[-1]
    return "_".join(name.split("_")[:2]) if split == "cityscapes" else row["image_path"].split("/")[3]


class S:  # Split-compatible container
    def __init__(self, fs, ts, fcm, tcm, cal, pooled):
        self.fit_scores, self.test_scores, self.fit_cm, self.test_cm = fs, ts, fcm, tcm
        self.cal, self.pooled, self.probe = cal, pooled, "tiny"
        self.fit_err = {lv: pixel_error(fcm[lv]) for lv in LEVELS}
        self.test_err = {lv: pixel_error(tcm[lv]) for lv in LEVELS}


def resplit(run):
    dump = json.loads((BASE / f"run_{run}_per_image.json").read_text())
    full, groups, info = {}, {}, {}
    for sp in SPLITS:
        d = dump[sp]
        n = len(d["fit_raw_scores"]) + len(d["test_raw_scores"])
        score = np.zeros(n)
        score[0::2], score[1::2] = d["fit_raw_scores"], d["test_raw_scores"]
        cm = {}
        for lv in LEVELS:
            f = np.asarray(d["fit_confusion_matrices"][lv], np.int64)
            t = np.asarray(d["test_confusion_matrices"][lv], np.int64)
            a = np.zeros((n, *f.shape[1:]), np.int64)
            a[0::2], a[1::2] = f, t
            cm[lv] = a
        rows = manifest(sp)
        assert len(rows) == n, (sp, len(rows), n)
        g = np.array([group_of(sp, r) for r in rows])
        sizes = {k: int((g == k).sum()) for k in np.unique(g)}
        fit_groups, hold_groups, nf, nh = [], [], 0, 0
        for k in sorted(sizes, key=lambda k: (-sizes[k], k)):
            if nf <= nh:
                fit_groups.append(k); nf += sizes[k]
            else:
                hold_groups.append(k); nh += sizes[k]
        fit = np.isin(g, fit_groups)
        full[sp] = (score, cm, fit)
        info[sp] = {"groups": len(sizes), "fit_images": int(fit.sum()), "heldout_images": int((~fit).sum()),
                    "fit_groups": len(fit_groups), "heldout_groups": len(hold_groups)}
    pooled_scores = np.concatenate([full[sp][0][full[sp][2]] for sp in SPLITS])
    pooled_err = {lv: np.concatenate([pixel_error(full[sp][1][lv][full[sp][2]]) for sp in SPLITS]) for lv in LEVELS}
    pooled = {lv: fit_risk_calibrator(pooled_scores, pooled_err[lv]) for lv in LEVELS}
    out = {}
    for sp in SPLITS:
        score, cm, fit = full[sp]
        fcm = {lv: cm[lv][fit] for lv in LEVELS}
        tcm = {lv: cm[lv][~fit] for lv in LEVELS}
        cal = {lv: fit_risk_calibrator(score[fit], pixel_error(fcm[lv])) for lv in LEVELS}
        out[sp] = S(score[fit], score[~fit], fcm, tcm, cal, pooled)
    grid = macro_quantile_grid([np.concatenate([out[sp].fit_err[lv] for lv in LEVELS]) for sp in SPLITS])
    return out, {sp: grid for sp in SPLITS}, info


def main() -> None:
    splits, grids, info = {}, {}, {}
    for run in RUNS:
        splits[run], grids[run], info[run] = resplit(run)
    res = {"_definition": __doc__, "split_info": info["a"]}
    # route-budget grid
    cells, th_rows = [], []
    for run in RUNS:
        for backend, cost in load_costs("median").items():
            budgets = sorted(set(cost.values()))
            for s, sp in splits[run].items():
                t = grids[run][s]
                pol = {"A": run_policy("A", sp, t, budgets, cost, sp.cal),
                       "A_hard": run_policy("A_hard", sp, t, budgets, cost, sp.cal),
                       "D": run_policy("D", sp, t, budgets, cost, sp.cal),
                       "static": static_best(sp, budgets, cost)}
                TH = threshold_policy(sp, budgets, cost)
                for b in budgets:
                    cells.append({"run": run, "split": s, **{k: v[b] for k, v in pol.items()}})
                    th_rows.append({"split": s, "D": pol["D"][b], "T_hard": TH[b]})
    res["route_grid"] = {
        "cells": len(cells),
        "A_violating_cells": int(sum(c["A"]["violation"] > 0 for c in cells)),
        "D_violating_cells": int(sum(c["D"]["violation"] > 0 for c in cells)),
        "D_vs_A_fair": compare(cells, "D", "A", "A"),
        "D_vs_A_hard": compare(cells, "D", "A_hard", None),
        "D_vs_T_hard": compare2(th_rows, "D", "T_hard"),
        "D_vs_static_route_cost": compare(cells, "D", "static", None),
    }
    # static-own-cost grids
    for name, (rc, sc) in configurations("median").items():
        if name not in ("E3_logits", "E1_explicit_float32"):
            continue
        budgets = sorted(set(rc.values()) | set(sc.values()))
        cheapest = min(rc.values())
        rows = []
        for run in RUNS:
            for s, sp in splits[run].items():
                D = run_policy("D", sp, grids[run][s], budgets, rc, sp.cal)
                ST = static_fair(sp, budgets, sc)
                for b in budgets:
                    rows.append({"split": s, "D": D[b], "static_S": ST[b], "feas": b >= cheapest - 1e-9})
        res[f"static_own_{name}"] = {"feasible": compare2([r for r in rows if r["feas"]], "D", "static_S"),
                                     "all": compare2(rows, "D", "static_S")}
    Path("reports/phaseB_sequence_split_20261004.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "_definition"}, indent=1, default=float))


if __name__ == "__main__":
    main()
