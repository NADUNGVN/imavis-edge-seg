"""Select reproducible qualitative examples from canonical held-out router artifacts.

This script does not need the licensed RGB datasets or PyTorch.  It maps the
per-image confusion matrices from the deployment-matched Run A dump back to the
versioned dataset manifests, then writes a selection manifest for the rendering
step.  The primary examples are the held-out images closest to the median
large-candidate pixel error in each split.  One additional, explicitly labelled
hardest ACDC example is retained for failure analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _pixel_error(confusion: list[list[int]]) -> float:
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(len(confusion)))
    return 0.0 if total == 0 else 1.0 - correct / total


def _image_id(image_path: str) -> str:
    name = Path(image_path).name
    for suffix in ("_leftImg8bit.png", "_rgb_anon.png"):
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return Path(name).stem


def _rows_for_split(
    split_name: str,
    cityscapes_rows: list[dict[str, str]],
    acdc_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    if split_name == "cityscapes":
        return cityscapes_rows
    condition = split_name.split("/", 1)[1]
    return [row for row in acdc_rows if row["image_path"].split("/")[1] == condition]


def _entry(
    split_name: str,
    selection_kind: str,
    heldout_position: int,
    row: dict[str, str],
    error: float,
    target_error: float,
) -> dict[str, Any]:
    return {
        "split": split_name,
        "condition": "clean" if split_name == "cityscapes" else split_name.split("/", 1)[1],
        "selection_kind": selection_kind,
        "heldout_position": heldout_position,
        "dataset_index": 2 * heldout_position + 1,
        "image_id": _image_id(row["image_path"]),
        "image_path": row["image_path"],
        "label_path": row["label_path"],
        "image_sha256": row["image_sha256"],
        "label_sha256": row["label_sha256"],
        "large_pixel_error": error,
        "selection_target_error": target_error,
    }


def build_selection(
    per_image_dump: dict[str, Any],
    cityscapes_rows: list[dict[str, str]],
    acdc_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    acdc_candidates: list[tuple[float, str, int, dict[str, str]]] = []

    for split_name in ("cityscapes", "acdc/fog", "acdc/night", "acdc/rain", "acdc/snow"):
        rows = _rows_for_split(split_name, cityscapes_rows, acdc_rows)
        heldout_rows = rows[1::2]
        matrices = per_image_dump[split_name]["test_confusion_matrices"]["large"]
        if len(heldout_rows) != len(matrices):
            raise ValueError(
                f"{split_name}: manifest has {len(heldout_rows)} held-out rows but artifact has "
                f"{len(matrices)} matrices"
            )
        errors = [_pixel_error(matrix) for matrix in matrices]
        target = median(errors)
        candidates = [
            (abs(error - target), row["image_path"], index, row, error)
            for index, (row, error) in enumerate(zip(heldout_rows, errors, strict=True))
        ]
        _, _, index, row, error = min(candidates)
        selections.append(_entry(split_name, "median_large_error", index, row, error, target))

        if split_name.startswith("acdc/"):
            acdc_candidates.extend(
                (error, split_name, index, row)
                for index, (row, error) in enumerate(zip(heldout_rows, errors, strict=True))
            )

    hardest_error, split_name, index, row = max(
        acdc_candidates,
        key=lambda item: (item[0], item[1], item[3]["image_path"]),
    )
    selections.append(
        _entry(
            split_name,
            "hardest_acdc_large_error",
            index,
            row,
            hardest_error,
            hardest_error,
        )
    )
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-image-dump",
        type=Path,
        default=Path("reports/router_deployment_matched_20260929/run_a_per_image.json"),
    )
    parser.add_argument(
        "--cityscapes-manifest",
        type=Path,
        default=Path("data/manifests/cityscapes_val.csv"),
    )
    parser.add_argument(
        "--acdc-manifest", type=Path, default=Path("data/manifests/acdc_val.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/qualitative/selection_manifest.json"),
    )
    args = parser.parse_args()

    with args.per_image_dump.open(encoding="utf-8") as stream:
        per_image_dump = json.load(stream)
    selections = build_selection(
        per_image_dump,
        _read_rows(args.cityscapes_manifest),
        _read_rows(args.acdc_manifest),
    )
    artifact = {
        "schema_version": 1,
        "source_per_image_dump": str(args.per_image_dump).replace("\\", "/"),
        "source_checkpoint_sha256": per_image_dump["_metadata"]["source_checkpoint_sha256"],
        "split_protocol": "alternating validation indices: even=fit, odd=held-out",
        "primary_rule": (
            "per split, select the held-out image nearest the median large-candidate pixel "
            "error; break ties by lexical image path"
        ),
        "failure_rule": "select the maximum large-candidate pixel error across ACDC held-out halves",
        "selections": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
