"""Freeze the held-out images used by the V20 qualitative figures (landscape Run A).

Runs locally: needs only the landscape per-image dump and the dataset manifests.
For each split it picks, from the held-out half:
  median   - large-candidate pixel error nearest the split median (typical case)
  gain     - largest tiny-minus-large pixel error (capacity matters most)
  easy     - smallest tiny-minus-large error among images with below-median tiny error
             (tiny is already enough)
  hard     - largest large-candidate pixel error (failure case)
Ties break by lexical image path. Output: paper/figures/qualitative/selection_v20.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

DUMP = Path("reports/landscape_20261004/run_a_per_image.json")
OUT = Path("paper/figures/qualitative/selection_v20.json")
SPLITS = ("cityscapes", "acdc/fog", "acdc/night", "acdc/rain", "acdc/snow")


def pixel_error(cm: list[list[int]]) -> float:
    tot = sum(map(sum, cm))
    return 1.0 - sum(cm[i][i] for i in range(len(cm))) / tot if tot else 0.0


def rows(split: str) -> list[dict[str, str]]:
    path = "data/manifests/cityscapes_val.csv" if split == "cityscapes" else "data/manifests/acdc_val.csv"
    with open(path, newline="", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    if split != "cityscapes":
        r = [x for x in r if x["image_path"].split("/")[1] == split.split("/")[1]]
    return r


def main() -> None:
    dump = json.loads(DUMP.read_text())
    meta = dump["_metadata"]
    out = []
    for split in SPLITS:
        cms = dump[split]["test_confusion_matrices"]
        err = {lv: [pixel_error(c) for c in cms[lv]] for lv in ("tiny", "small", "medium", "large")}
        manifest = rows(split)
        n = len(err["large"])
        held = [manifest[2 * i + 1] for i in range(n)]
        key = lambda i: held[i]["image_path"]
        med = median(err["large"])
        tmed = median(err["tiny"])
        gain = [err["tiny"][i] - err["large"][i] for i in range(n)]
        picks = {
            "median": min(range(n), key=lambda i: (abs(err["large"][i] - med), key(i))),
            "gain": max(range(n), key=lambda i: (gain[i], key(i))),
            "easy": min((i for i in range(n) if err["tiny"][i] <= tmed), key=lambda i: (gain[i], key(i))),
            "hard": max(range(n), key=lambda i: (err["large"][i], key(i))),
        }
        for kind, i in picks.items():
            row = held[i]
            out.append({
                "split": split, "kind": kind, "heldout_position": i, "dataset_index": 2 * i + 1,
                "image_path": row["image_path"], "label_path": row["label_path"],
                "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
                "pixel_error": {lv: err[lv][i] for lv in err},
            })
    OUT.write_text(json.dumps({
        "schema_version": 2,
        "source_per_image_dump": str(DUMP).replace("\\", "/"),
        "source_checkpoint": meta["source_checkpoint"],
        "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
        "split_protocol": meta["split_protocol"],
        "selections": out,
    }, indent=2))
    print(f"wrote {OUT} ({len(out)} entries, {len({e['image_path'] for e in out})} unique images)")


if __name__ == "__main__":
    main()
