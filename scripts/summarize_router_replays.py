"""Summarize deployment-matched router replays without treating cells as samples.

The script computes descriptive counts only.  A fair cell is an operating point
where baseline A has zero held-out budget violations.  Win/tie/loss and mean
quality deltas use fair cells; violation counts use the full grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _cell_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    metadata = payload.get("_metadata", {})
    fair = wins = ties = losses = a_violating = d_violating = total = 0
    deltas: list[float] = []

    for split_name, split in payload.items():
        if split_name.startswith("_"):
            continue
        e2e = split["e2e_aware"]
        policy_a = e2e["calibrated_risk"]
        policy_d = e2e["risk_latency_constrained"]
        if set(policy_a) != set(policy_d):
            raise ValueError(f"{path}: A/D budget grids differ for {split_name}")
        for budget in policy_a:
            total += 1
            a = policy_a[budget]
            d = policy_d[budget]
            if float(a["violation_rate"]) > 0.0:
                a_violating += 1
            if float(d["violation_rate"]) > 0.0:
                d_violating += 1
            if float(a["violation_rate"]) != 0.0:
                continue
            fair += 1
            delta = float(d["achieved_miou"]) - float(a["achieved_miou"])
            deltas.append(delta)
            if abs(delta) <= 1e-12:
                ties += 1
            elif delta > 0.0:
                wins += 1
            else:
                losses += 1

    return {
        "artifact": str(path),
        "run_label": metadata.get("run_label"),
        "backend": metadata.get("device_id"),
        "total_cells": total,
        "fair_cells": fair,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_delta_miou_fair": sum(deltas) / len(deltas) if deltas else None,
        "a_violating_cells": a_violating,
        "d_violating_cells": d_violating,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fair = sum(int(row["fair_cells"]) for row in rows)
    weighted_delta = sum(
        float(row["mean_delta_miou_fair"]) * int(row["fair_cells"])
        for row in rows
        if row["mean_delta_miou_fair"] is not None
    )
    return {
        "total_cells": sum(int(row["total_cells"]) for row in rows),
        "fair_cells": fair,
        "wins": sum(int(row["wins"]) for row in rows),
        "ties": sum(int(row["ties"]) for row in rows),
        "losses": sum(int(row["losses"]) for row in rows),
        "mean_delta_miou_fair": weighted_delta / fair if fair else None,
        "a_violating_cells": sum(int(row["a_violating_cells"]) for row in rows),
        "d_violating_cells": sum(int(row["d_violating_cells"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = [_cell_summary(path) for path in args.replay]
    by_backend: dict[str, dict[str, Any]] = {}
    for backend in sorted({str(row["backend"]) for row in rows}):
        by_backend[backend] = _aggregate([row for row in rows if str(row["backend"]) == backend])
    result = {
        "definition": {
            "fair_cell": "baseline A held-out violation_rate equals zero",
            "comparison": "D minus A achieved mIoU at the same measured-cost budget",
            "statistical_boundary": "operating cells are correlated descriptive evaluations, not independent samples",
        },
        "per_run_backend": rows,
        "by_backend": by_backend,
        "overall": _aggregate(rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
