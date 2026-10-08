"""Stratified, non-cherry-picked image selection for the V20 qualitative figure.

Rule (fixed before looking at any image): run policy D of Run A on E3 with the
median same-harness cost table at the budget equal to the large-route cost, pooled
over the five held-out splits. For each capacity that D selects (tiny/small/medium/
large), take the routed image whose own selected-capacity pixel error is closest to
the median error of that route class; ties break by lexical image path. Classes with
no routed image are reported, not filled.

Output: paper/figures/qualitative/selection_v20_stratified.json (same schema as
selection_v20.json, readable by scripts/dump_qualitative_assets_v20.py).

  PYTHONPATH="src;scripts" PACE_ROUTER_BASE=reports/landscape_20261004 PACE_TAG=20261004 \
  PACE_ROUTE_COSTS=same_harness python paper/figures/scripts/select_qualitative_v20_stratified.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

from router_review_analyses import Split, load_costs, run_policy

BASE = Path("reports/landscape_20261004")
OUT = Path("paper/figures/qualitative/selection_v20_stratified.json")
SPLITS = ("cityscapes", "acdc/fog", "acdc/night", "acdc/rain", "acdc/snow")
LEVELS = ("tiny", "small", "medium", "large")


def pixel_error(cm) -> float:
    tot = sum(map(sum, cm))
    return 1.0 - sum(cm[i][i] for i in range(len(cm))) / tot if tot else 0.0


def manifest_rows(split: str) -> list[dict[str, str]]:
    path = "data/manifests/cityscapes_val.csv" if split == "cityscapes" else "data/manifests/acdc_val.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split != "cityscapes":
        rows = [r for r in rows if r["image_path"].split("/")[1] == split.split("/")[1]]
    return rows


def main() -> None:
    dump = json.loads((BASE / "run_a_per_image.json").read_text())
    ev = json.loads((BASE / "run_a_evaluation.json").read_text())
    cost = load_costs("median")["E3"]
    budget = cost["large"]
    pool = []
    for split in SPLITS:
        sp = Split(dump[split])
        res = run_policy("D", sp, ev[split]["risk_target_grid"], [budget], cost, sp.cal)[budget]
        rows = manifest_rows(split)
        cms = dump[split]["test_confusion_matrices"]
        for i, ch in enumerate(res["choice"]):
            pool.append({"split": split, "i": i, "choice": ch, "row": rows[2 * i + 1],
                         "err": {lv: pixel_error(cms[lv][i]) for lv in LEVELS}})
    counts = {lv: sum(p["choice"] == lv for p in pool) for lv in LEVELS}
    out = []
    for lv in LEVELS:
        cls = [p for p in pool if p["choice"] == lv]
        if not cls:
            continue
        med = median(p["err"][lv] for p in cls)
        p = min(cls, key=lambda q: (abs(q["err"][lv] - med), q["row"]["image_path"]))
        out.append({"split": p["split"], "kind": f"routed_{lv}", "heldout_position": p["i"],
                    "dataset_index": 2 * p["i"] + 1, "image_path": p["row"]["image_path"],
                    "label_path": p["row"]["label_path"], "image_sha256": p["row"]["image_sha256"],
                    "label_sha256": p["row"]["label_sha256"], "pixel_error": p["err"], "routed_to": lv})
    meta = dump["_metadata"]
    OUT.write_text(json.dumps({
        "schema_version": 2, "rule": __doc__.split("Output:")[0].strip(),
        "budget_ms_E3": budget, "route_class_counts": counts,
        "source_per_image_dump": str(BASE / "run_a_per_image.json").replace("\\", "/"),
        "source_checkpoint": meta["source_checkpoint"], "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
        "split_protocol": meta["split_protocol"], "selections": out}, indent=2))
    print("route-class counts", counts)
    for e in out:
        print(e["routed_to"], e["split"], e["image_path"])


if __name__ == "__main__":
    main()
