"""Routing replay with ON-DEVICE predictions (review round 2, 2026-10-03).

Run A only (the compiled engines were built from the Run A checkpoint). For each
device, the held-out per-image confusion matrices and the tiny-probe entropy are
replaced by those of the compiled engines (reports/compiled_eval_<dev>_20261003_
per_image.npz), while the calibrators, risk-target grid and fit-half operating-point
selection stay exactly as fitted on PyTorch predictions -- which is what a deployment
would do. Costs are the same-harness tables (route and direct static) of that device.

Reports, for PyTorch-replay vs on-device-replay: D vs T-hard, D vs A-hard, D vs best
feasible static, and the mean-cost comparison against static mixing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from router_review_analyses import BASE, LEVELS, Split, pixel_error, run_policy
from router_review_analyses_v2 import compare, mixture_gain, static_fair, threshold_policy
from router_same_harness_analysis import tables_from

DEVICES = {
    "E3": ("reports/compiled_eval_E3_20261003_per_image.npz", "reports/static_vs_route_E3_20261003.json", "|logits"),
    "E1": ("reports/compiled_eval_E1_20261003_per_image.npz", "reports/static_vs_route_E1_explicit_float32_20261003.json", ""),
}
SPLIT_KEY = {"cityscapes": "cityscapes", "acdc/fog": "acdc_fog", "acdc/night": "acdc_night",
             "acdc/rain": "acdc_rain", "acdc/snow": "acdc_snow"}


def ondevice_split(sp: Split, npz, key: str) -> Split:
    new = copy.copy(sp)
    new.test_cm = {lv: npz[f"{key}__{lv}"].astype(np.int64) for lv in LEVELS}
    new.test_err = {lv: pixel_error(new.test_cm[lv]) for lv in LEVELS}
    new.test_scores = npz[f"{key}__tiny_entropy"].astype(np.float64)
    for lv in LEVELS:
        if len(new.test_cm[lv]) != len(sp.test_scores):
            raise SystemExit(f"held-out size mismatch {key} {lv}")
    return new


def evaluate(splits: dict[str, Split], grids: dict[str, list[float]], rc: dict, sc: dict) -> dict:
    budgets = sorted(set(rc.values()) | set(sc.values()))
    rows = []
    for name, sp in splits.items():
        D = run_policy("D", sp, grids[name], budgets, rc, sp.cal)
        AH = run_policy("A_hard", sp, grids[name], budgets, rc, sp.cal)
        TH = threshold_policy(sp, budgets, rc)
        ST = static_fair(sp, budgets, sc)
        for b in budgets:
            rows.append({"split": name, "D": D[b], "A_hard": AH[b], "T_hard": TH[b], "static_S": ST[b]})
    import router_review_analyses_v2 as v2

    saved, v2.RUNS = v2.RUNS, ("a",)  # Run A only for the on-device comparison
    try:
        mix = mixture_gain({"a": splits}, {"dev": rc}, {"dev": sc})
    finally:
        v2.RUNS = saved
    return {"D_vs_T_hard": compare(rows, "D", "T_hard"), "D_vs_A_hard": compare(rows, "D", "A_hard"),
            "D_vs_static": compare(rows, "D", "static_S"),
            "D_violating_cells": int(sum(r["D"]["violation"] > 0 for r in rows)),
            "mixture_gain_points": mix["mean_gain_points"], "mixture_share_above": mix["share_above"],
            "mean_D_miou_points": float(np.mean([r["D"]["miou"] for r in rows]) * 100)}


def main() -> None:
    dump = json.loads((BASE / "run_a_per_image.json").read_text())
    ev = json.loads((BASE / "run_a_evaluation.json").read_text())
    torch_splits = {k: Split(v) for k, v in dump.items() if not k.startswith("_")}
    grids = {k: ev[k]["risk_target_grid"] for k in torch_splits}
    out = {}
    for dev, (npz_path, cost_path, suffix) in DEVICES.items():
        npz = np.load(npz_path)
        rc, sc = tables_from(Path(cost_path), "median", suffix)
        dev_splits = {k: ondevice_split(sp, npz, SPLIT_KEY[k]) for k, sp in torch_splits.items()}
        out[dev] = {"torch_replay": evaluate(torch_splits, grids, rc, sc),
                    "ondevice_replay": evaluate(dev_splits, grids, rc, sc)}
        for mode in ("torch_replay", "ondevice_replay"):
            r = out[dev][mode]
            print(f"{dev} {mode:15s} D mIoU {r['mean_D_miou_points']:.2f} | D-T {r['D_vs_T_hard']['mean_delta_points']:+.2f} "
                  f"| D-Ahard {r['D_vs_A_hard']['mean_delta_points']:+.2f} | D-static {r['D_vs_static']['mean_delta_points']:+.2f} "
                  f"| viol {r['D_violating_cells']} | mix {r['mixture_gain_points']:+.2f} ({r['mixture_share_above']:.0%})", flush=True)
    Path("reports/router_ondevice_replay_20261003.json").write_text(json.dumps(out, indent=2))
    print("wrote reports/router_ondevice_replay_20261003.json")


if __name__ == "__main__":
    main()
