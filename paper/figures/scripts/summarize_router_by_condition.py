"""Write descriptive configured/pooled router comparisons by condition."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

RUNS = ("a", "b", "c")
BACKENDS = ("e1", "e3")
POLICIES = {
    "configured": "risk_latency_constrained",
    "pooled": "pooled_risk_latency_constrained",
}
CONDITION_LABELS = {
    "cityscapes": "Cityscapes",
    "acdc/fog": "ACDC/Fog",
    "acdc/night": "ACDC/Night",
    "acdc/rain": "ACDC/Rain",
    "acdc/snow": "ACDC/Snow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[3])
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).parents[2] / "tables" / "router_by_condition.csv",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path(__file__).parents[2] / "tables" / "router_by_condition_table.tex",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def result_label(delta: float, tolerance: float = 1e-12) -> str:
    if delta > tolerance:
        return "wins"
    if delta < -tolerance:
        return "losses"
    return "ties"


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    evidence = repo / "reports" / "router_deployment_matched_20260929"
    accum: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "fair": 0,
            "a_violations": 0,
            "configured_violations": 0,
            "pooled_violations": 0,
            "configured_deltas": [],
            "pooled_deltas": [],
            "configured_counts": {"wins": 0, "ties": 0, "losses": 0},
            "pooled_counts": {"wins": 0, "ties": 0, "losses": 0},
        }
    )

    for run in RUNS:
        for backend in BACKENDS:
            payload = read_json(evidence / f"run_{run}_{backend}_replay.json")
            for split_name, split in payload.items():
                if split_name.startswith("_"):
                    continue
                row = accum[split_name]
                policies = split["e2e_aware"]
                baseline = policies["calibrated_risk"]
                for budget, point_a in baseline.items():
                    row["total"] += 1
                    a_violation = float(point_a["violation_rate"]) > 0.0
                    row["a_violations"] += int(a_violation)
                    for label, policy_key in POLICIES.items():
                        point_d = policies[policy_key][budget]
                        row[f"{label}_violations"] += int(
                            float(point_d["violation_rate"]) > 0.0
                        )
                    if a_violation:
                        continue
                    row["fair"] += 1
                    for label, policy_key in POLICIES.items():
                        point_d = policies[policy_key][budget]
                        delta = float(point_d["achieved_miou"]) - float(
                            point_a["achieved_miou"]
                        )
                        row[f"{label}_deltas"].append(delta)
                        row[f"{label}_counts"][result_label(delta)] += 1

    rows = []
    for split_name in CONDITION_LABELS:
        values = accum[split_name]
        configured = values["configured_counts"]
        pooled = values["pooled_counts"]
        rows.append(
            {
                "condition": CONDITION_LABELS[split_name],
                "total_cells": values["total"],
                "fair_cells": values["fair"],
                "configured_w_t_l": f"{configured['wins']}/{configured['ties']}/{configured['losses']}",
                "configured_mean_delta_miou": sum(values["configured_deltas"])
                / len(values["configured_deltas"]),
                "pooled_w_t_l": f"{pooled['wins']}/{pooled['ties']}/{pooled['losses']}",
                "pooled_mean_delta_miou": sum(values["pooled_deltas"])
                / len(values["pooled_deltas"]),
                "a_violating_cells": values["a_violations"],
                "configured_d_violating_cells": values["configured_violations"],
                "pooled_d_violating_cells": values["pooled_violations"],
            }
        )

    output = args.output_csv.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")

    tex_output = args.output_tex.resolve()
    tex_output.parent.mkdir(parents=True, exist_ok=True)
    tex_lines = []
    for row in rows:
        tex_lines.append(
            f"{row['condition']} & {row['fair_cells']}/{row['total_cells']} & "
            f"{row['configured_w_t_l']} & {row['configured_mean_delta_miou']:+.4f} & "
            f"{row['pooled_w_t_l']} & {row['pooled_mean_delta_miou']:+.4f} & "
            f"{row['a_violating_cells']}/{row['total_cells']} & "
            f"{row['configured_d_violating_cells']}/{row['total_cells']} \\\\"
        )
    table_lines = [
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        (
            r"Condition & Fair & Config. W/T/L & Config. $\Delta$ & "
            r"Pooled W/T/L & Pooled $\Delta$ & A viol. & D viol. \\"
        ),
        r"\midrule",
        *tex_lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    tex_output.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"Wrote {tex_output}")


if __name__ == "__main__":
    main()
