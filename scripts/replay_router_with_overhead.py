"""Offline replay of the router progressive ablation against real end-to-end
overhead (`docs/COORDINATION_LOG.md` open thread #2, closing requirement 2),
protocol locked with Codex 2026-09-22: aggregated routing-distribution reweighting
is not sufficient to close this requirement -- the policy's own budget-feasibility
decision must be re-run against the new, real latency table, not just relabeled
after the fact.

Inputs:
- `--per-image-dump`: `scripts/evaluate_router.py --dump-per-image`'s output (raw
  probe scores + per-level ground-truth confusion matrices + fitted calibrators,
  fit-half AND held-out, per split).
- `--overhead-json`: `scripts/measure_router_overhead.py`'s output (real warm
  end-to-end latency per route class, GPU-kernel backend).
- `--original-results-json`: `scripts/evaluate_router.py --output-json`'s output for
  the SAME checkpoint/splits (only used for `risk_target_grid`, which is
  device-independent and already locked -- not re-derived here).
- `--lookup-table`/`--device-id`/`--backend`: the original LUT-only latencies, for
  the post-hoc replay only.

No re-inference: every number here is pure post-processing over the dump's cached
per-image data, since route latency depends only on the destination level, not
image content.

Two replay modes, kept separate per Codex's instruction (never blend them):
- **post-hoc**: the policy's decision is recomputed EXACTLY as originally run (LUT
  latencies, LUT budget grid, the already-chosen risk_target per operating point) --
  only the *scoring* (mean/p95 latency, violation rate) substitutes real e2e latency
  for the chosen level. Shows how wrong the old LUT-only latency model was, nothing
  more -- the policy itself never saw real latency.
- **e2e-aware**: the policy's decision is recomputed using real e2e latencies as
  BOTH the ranking/budget-check input AND the new device-budget grid, with the
  representative operating point re-selected via `select_budget_matched_operating_
  point` on FIT-HALF stats (locked before touching held-out, same discipline as the
  original run). This is the real deployment result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from imavis_edge_seg.config import ElasticityLevel, RouterConfig
from imavis_edge_seg.router.calibrator import RiskCalibrator
from imavis_edge_seg.router.grid import select_budget_matched_operating_point
from imavis_edge_seg.router.policy import select_level
from imavis_edge_seg.search.pareto import ParetoPoint

LEVELS: list[ElasticityLevel] = ["tiny", "small", "medium", "large"]
_ReplayStrategy = Literal["calibrated_risk", "risk_latency_constrained"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def per_image_error_from_confusion(matrix: list[list[int]]) -> float:
    """1 - pixel accuracy, derived from a confusion matrix -- identical to
    `router.observed_error.compute_per_image_error`'s definition (both exclude
    IGNORE_INDEX pixels; the confusion matrix already excludes them by
    construction)."""
    arr = np.asarray(matrix, dtype=np.int64)
    total = arr.sum()
    if total == 0:
        return 0.0
    return float(1.0 - arr.trace() / total)


def miou_from_confusion_matrices(matrices: list[list[list[int]]]) -> float:
    """Sums per-image confusion matrices and computes mIoU once -- identical to
    `evaluation.metrics.ConfusionMatrixAccumulator`, without needing torch."""
    total = np.zeros_like(np.asarray(matrices[0], dtype=np.int64))
    for m in matrices:
        total += np.asarray(m, dtype=np.int64)
    tp = np.diagonal(total).astype(np.float64)
    predicted = total.sum(axis=0).astype(np.float64)
    actual = total.sum(axis=1).astype(np.float64)
    union = predicted + actual - tp
    per_class_iou = np.full_like(tp, np.nan, dtype=np.float64)
    np.divide(tp, union, out=per_class_iou, where=union > 0)
    return float(np.nanmean(per_class_iou))


def load_lut_latency(
    lookup_table: Path, device_id: str, backend: str, latency_field: str
) -> dict[ElasticityLevel, float]:
    with lookup_table.open() as f:
        rows = list(csv.DictReader(f))
    lut: dict[ElasticityLevel, float] = {}
    for level in LEVELS:
        matches = [
            r
            for r in rows
            if r["device_id"] == device_id and r["backend"] == backend and r["level"] == level
        ]
        if not matches or not matches[0].get(latency_field):
            raise ValueError(f"no {device_id}/{backend} lookup-table row for level {level!r}")
        lut[level] = float(matches[0][latency_field])
    return lut


def load_e2e_latency(
    overhead_json: dict[str, Any], entropy_backend: str = "gpu"
) -> dict[ElasticityLevel, float]:
    """Real warm end-to-end latency per level, keyed by destination level (probe is
    always "tiny" in the overhead harness). `entropy_backend`: "gpu" (E3's GPU-kernel
    backend, the representative/optimized path -- see
    `reports/router_overhead_v1_20260922.md`) or "numpy" (the only backend that exists
    on E1/Hailo -- no GPU-resident entropy kernel is possible on that architecture, see
    `scripts/measure_router_overhead_hailo.py`'s module docstring)."""
    warm = overhead_json["warm"][entropy_backend]
    e2e: dict[ElasticityLevel, float] = {}
    for level in LEVELS:
        route_key = f"tiny->{level}"
        e2e[level] = warm[route_key]["median_ms"]
    return e2e


def decide_levels(
    strategy: _ReplayStrategy,
    risk_target: float,
    latency_budget_ms: float | None,
    latencies_ms: dict[ElasticityLevel, float],
    raw_scores: list[float],
    probe_calibrator: RiskCalibrator,
    per_level_calibrators: dict[ElasticityLevel, RiskCalibrator],
) -> list[ElasticityLevel]:
    """Recomputes the per-image chosen level for `strategy` ("calibrated_risk" or
    "risk_latency_constrained"), given a specific latency table -- the actual policy
    decision, not just a relabeling of an already-made choice."""
    candidates = [
        ParetoPoint(
            level=level,
            device_id="replay",
            backend="tensorrt_gpu",
            precision="fp16",
            latency_ms=lat,
            miou=0.0,
            dataset="n/a",
        )
        for level, lat in latencies_ms.items()
    ]
    config = RouterConfig(
        strategy=strategy, risk_target=risk_target, latency_budget_ms=latency_budget_ms
    )
    chosen: list[ElasticityLevel] = []
    for s in raw_scores:
        if strategy == "calibrated_risk":
            calibrated_risk = probe_calibrator.predict(s)
            chosen.append(select_level(candidates, config, calibrated_risk=calibrated_risk))
        else:
            per_level_risk = {level: cal.predict(s) for level, cal in per_level_calibrators.items()}
            chosen.append(select_level(candidates, config, per_level_risk=per_level_risk))
    return chosen


def score(
    chosen: list[ElasticityLevel],
    confusion_matrices: dict[ElasticityLevel, list[list[list[int]]]],
    latencies_ms: dict[ElasticityLevel, float],
) -> dict[str, Any]:
    matrices = [confusion_matrices[level][i] for i, level in enumerate(chosen)]
    lat_list = [latencies_ms[level] for level in chosen]
    return {
        "achieved_miou": miou_from_confusion_matrices(matrices),
        "mean_latency_ms": float(np.mean(lat_list)),
        "p95_latency_ms": float(np.percentile(lat_list, 95)),
        "chosen_latencies_ms": lat_list,
    }


def replay_post_hoc(
    split_dump: dict[str, Any],
    original_split_result: dict[str, Any],
    lut_latency: dict[ElasticityLevel, float],
    e2e_latency: dict[ElasticityLevel, float],
) -> dict[str, Any]:
    """Recomputes the ORIGINAL policy decision exactly (LUT latencies/budget, the
    already-chosen risk_target per operating point), then scores it with real e2e
    latency instead of LUT-only latency."""
    per_level_calibrators = {
        level: RiskCalibrator.from_dict(d)
        for level, d in split_dump["per_level_calibrators"].items()
    }
    pooled_calibrators = {
        level: RiskCalibrator.from_dict(d)
        for level, d in split_dump.get("pooled_per_level_calibrators", {}).items()
    }
    probe_calibrator = per_level_calibrators[split_dump["probe_level"]]
    test_raw_scores = split_dump["test_raw_scores"]
    test_confusion = split_dump["test_confusion_matrices"]

    strategy_specs: list[tuple[str, _ReplayStrategy, dict[ElasticityLevel, RiskCalibrator]]] = [
        ("calibrated_risk", "calibrated_risk", per_level_calibrators),
        ("risk_latency_constrained", "risk_latency_constrained", per_level_calibrators),
    ]
    if (
        pooled_calibrators
        and "pooled_risk_latency_constrained" in original_split_result["at_budget"]
    ):
        strategy_specs.append(
            ("pooled_risk_latency_constrained", "risk_latency_constrained", pooled_calibrators)
        )
    results: dict[str, dict[str, Any]] = {name: {} for name, _, _ in strategy_specs}
    for result_name, policy_strategy, calibrators in strategy_specs:
        at_budget = original_split_result["at_budget"][result_name]
        for budget_str, point in at_budget.items():
            budget = float(budget_str)
            risk_target = point["risk_target"]
            latency_budget = budget if policy_strategy == "risk_latency_constrained" else None
            chosen = decide_levels(
                policy_strategy,
                risk_target,
                latency_budget,
                lut_latency,
                test_raw_scores,
                probe_calibrator,
                calibrators,
            )
            scored = score(chosen, test_confusion, e2e_latency)
            scored["violation_rate"] = float(
                np.mean([lat > budget for lat in scored["chosen_latencies_ms"]])
            )
            scored["risk_target"] = risk_target
            scored["original_budget_ms"] = budget
            results[result_name][budget_str] = scored
    return results


def replay_e2e_aware(
    split_dump: dict[str, Any],
    risk_target_grid: list[float],
    e2e_budget_grid: list[float],
    e2e_latency: dict[ElasticityLevel, float],
) -> dict[str, Any]:
    """Recomputes the policy decision using real e2e latency as BOTH the
    ranking/budget-check input and the new device-budget grid. The representative
    operating point per budget is re-selected via `select_budget_matched_operating_
    point` using FIT-HALF stats only, before held-out is touched -- the same
    discipline the original run used."""
    per_level_calibrators = {
        level: RiskCalibrator.from_dict(d)
        for level, d in split_dump["per_level_calibrators"].items()
    }
    pooled_calibrators = {
        level: RiskCalibrator.from_dict(d)
        for level, d in split_dump.get("pooled_per_level_calibrators", {}).items()
    }
    probe_calibrator = per_level_calibrators[split_dump["probe_level"]]
    fit_raw_scores = split_dump["fit_raw_scores"]
    fit_confusion = split_dump["fit_confusion_matrices"]
    test_raw_scores = split_dump["test_raw_scores"]
    test_confusion = split_dump["test_confusion_matrices"]

    strategy_specs: list[tuple[str, _ReplayStrategy, dict[ElasticityLevel, RiskCalibrator]]] = [
        ("calibrated_risk", "calibrated_risk", per_level_calibrators),
        ("risk_latency_constrained", "risk_latency_constrained", per_level_calibrators),
    ]
    if pooled_calibrators:
        strategy_specs.append(
            ("pooled_risk_latency_constrained", "risk_latency_constrained", pooled_calibrators)
        )
    results: dict[str, dict[str, Any]] = {name: {} for name, _, _ in strategy_specs}
    for result_name, policy_strategy, calibrators in strategy_specs:
        # Build the frontier (one point per risk_target) on fit-half AND held-out,
        # under the NEW e2e latency table.
        frontier = []
        for risk_target in risk_target_grid:
            # For D, the "grid" risk_target is crossed with EVERY e2e budget below;
            # A doesn't take a latency_budget_ms at all -- pass None throughout, its
            # own decision only depends on the e2e latencies used for ranking.
            latency_budget_for_decision = (
                None if policy_strategy == "calibrated_risk" else max(e2e_budget_grid)
            )
            fit_chosen = decide_levels(
                policy_strategy,
                risk_target,
                latency_budget_for_decision,
                e2e_latency,
                fit_raw_scores,
                probe_calibrator,
                calibrators,
            )
            test_chosen = decide_levels(
                policy_strategy,
                risk_target,
                latency_budget_for_decision,
                e2e_latency,
                test_raw_scores,
                probe_calibrator,
                calibrators,
            )
            fit_scored = score(fit_chosen, fit_confusion, e2e_latency)
            test_scored = score(test_chosen, test_confusion, e2e_latency)
            frontier.append((risk_target, fit_scored, test_scored))

        for budget in e2e_budget_grid:
            if policy_strategy == "risk_latency_constrained":
                # D's decision genuinely depends on the budget -- recompute per budget.
                points = []
                for risk_target in risk_target_grid:
                    fit_chosen = decide_levels(
                        policy_strategy,
                        risk_target,
                        budget,
                        e2e_latency,
                        fit_raw_scores,
                        probe_calibrator,
                        calibrators,
                    )
                    test_chosen = decide_levels(
                        policy_strategy,
                        risk_target,
                        budget,
                        e2e_latency,
                        test_raw_scores,
                        probe_calibrator,
                        calibrators,
                    )
                    fit_scored = score(fit_chosen, fit_confusion, e2e_latency)
                    test_scored = score(test_chosen, test_confusion, e2e_latency)
                    points.append((risk_target, fit_scored, test_scored))
            else:
                points = frontier

            operating_points = [(p[1]["mean_latency_ms"], p[1]["achieved_miou"], p) for p in points]
            chosen_point, feasible = select_budget_matched_operating_point(operating_points, budget)
            risk_target, fit_scored, test_scored = cast(
                tuple[float, dict[str, Any], dict[str, Any]], chosen_point
            )
            results[result_name][str(budget)] = {
                "risk_target": risk_target,
                "feasible": feasible,
                "achieved_miou": test_scored["achieved_miou"],
                "mean_latency_ms": test_scored["mean_latency_ms"],
                "p95_latency_ms": test_scored["p95_latency_ms"],
                "violation_rate": float(
                    np.mean([lat > budget for lat in test_scored["chosen_latencies_ms"]])
                ),
            }
    return results


def replay_oracle(
    split_dump: dict[str, Any],
    e2e_budget_grid: list[float],
    e2e_latency: dict[ElasticityLevel, float],
) -> dict[str, Any]:
    """Budget-matched, per-image oracle under real e2e latency: for each image and
    budget, the candidate with the lowest ground-truth per-image error among those
    within budget."""
    test_confusion = split_dump["test_confusion_matrices"]
    n = len(split_dump["test_raw_scores"])
    per_level_error = {
        level: [per_image_error_from_confusion(m) for m in test_confusion[level]]
        for level in LEVELS
    }

    results: dict[str, Any] = {}
    for budget in e2e_budget_grid:
        in_budget = [level for level in LEVELS if e2e_latency[level] <= budget] or [
            min(LEVELS, key=lambda lv: e2e_latency[lv])
        ]
        chosen: list[ElasticityLevel] = [
            min(in_budget, key=lambda lv: per_level_error[lv][i]) for i in range(n)
        ]
        scored = score(chosen, test_confusion, e2e_latency)
        scored["violation_rate"] = float(
            np.mean([lat > budget for lat in scored["chosen_latencies_ms"]])
        )
        results[str(budget)] = scored
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image-dump", type=Path, required=True)
    parser.add_argument("--overhead-json", type=Path, required=True)
    parser.add_argument("--original-results-json", type=Path, required=True)
    parser.add_argument(
        "--lookup-table", type=Path, default=Path("outputs/benchmark_lookup_table.csv")
    )
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--latency-field", default="end_to_end_p95_ms")
    parser.add_argument(
        "--entropy-backend", default="gpu", help='"gpu" (E3) or "numpy" (only option on E1/Hailo)'
    )
    parser.add_argument(
        "--run-label", default=None, help="manuscript-facing run label stored in artifact metadata"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    dump = json.loads(args.per_image_dump.read_text())
    overhead = json.loads(args.overhead_json.read_text())
    original = json.loads(args.original_results_json.read_text())
    lut_latency = load_lut_latency(
        args.lookup_table, args.device_id, args.backend, args.latency_field
    )
    e2e_latency = load_e2e_latency(overhead, args.entropy_backend)
    e2e_budget_grid = sorted(set(e2e_latency.values()))

    print(f"LUT-only latency: {lut_latency}")
    print(f"real e2e latency ({args.entropy_backend} backend, warm): {e2e_latency}")
    print(f"new e2e-based budget grid: {e2e_budget_grid}")

    results: dict[str, Any] = {
        "_metadata": {
            "schema_version": 2,
            "run_label": args.run_label or dump.get("_metadata", {}).get("run_label"),
            "source_per_image_dump": str(args.per_image_dump),
            "source_per_image_dump_sha256": _sha256_file(args.per_image_dump),
            "source_results": str(args.original_results_json),
            "source_results_sha256": _sha256_file(args.original_results_json),
            "overhead_measurements": str(args.overhead_json),
            "overhead_measurements_sha256": _sha256_file(args.overhead_json),
            "lookup_table": str(args.lookup_table),
            "lookup_table_sha256": _sha256_file(args.lookup_table),
            "device_id": args.device_id,
            "backend": args.backend,
            "entropy_backend": args.entropy_backend,
            "latency_field": args.latency_field,
            "budget_grid_ms": e2e_budget_grid,
            "policies": [
                "calibrated_risk",
                "risk_latency_constrained",
                "pooled_risk_latency_constrained",
                "oracle",
            ],
            "risk_feature": dump.get("_metadata", {}).get("risk_feature"),
            "code_commit": _git_commit(),
        }
    }
    for split_name, split_dump in dump.items():
        if split_name.startswith("_"):
            continue
        risk_target_grid = original[split_name]["risk_target_grid"]
        print(f"replaying {split_name}...")
        post_hoc = replay_post_hoc(split_dump, original[split_name], lut_latency, e2e_latency)
        e2e_aware = replay_e2e_aware(split_dump, risk_target_grid, e2e_budget_grid, e2e_latency)
        oracle = replay_oracle(split_dump, e2e_budget_grid, e2e_latency)
        results[split_name] = {"post_hoc": post_hoc, "e2e_aware": e2e_aware, "oracle": oracle}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
