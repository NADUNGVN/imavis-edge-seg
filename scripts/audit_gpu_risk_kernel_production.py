"""Second half of Codex's mandatory audit requirement (2026-09-29 review): compare
E3's real GPU-kernel risk scores against the ACTUAL production
`imavis_edge_seg.router.risk_probe.compute_risk_score` (a real import, not a numpy
transcription -- closes Codex's "chua hoan toan tuong duong ... PyTorch-reference
audit" gap in the first audit), on REAL captured TensorRT engine outputs (not
synthetic logits -- catches layout/dtype/buffer-indexing errors a synthetic sweep
could miss), and re-checks A/D decision agreement using the real
`router.calibrator.RiskCalibrator` and `router.policy` decision functions.

Reads `reports/real_outputs/{level}.npz` (from
`scripts/capture_real_engine_outputs.py`, run on E3): `logits` (N, flat_size) float32
-- the exact host-copied bytes of a real (untrained-weight, legitimate for a pure
numerics check per this project's established precedent) TensorRT engine's output --
and `gpu_risk_score` (N,) -- the real deployed GPU-kernel path's score for the SAME
sample (computed via `infer_no_copy` + `risk_score_gpu`, reading `device_out` directly,
not the host-copied array saved here -- so this checks genuine cross-implementation
agreement, not a self-comparison).

Usage:
    uv run python scripts/audit_gpu_risk_kernel_production.py --output-json reports/audit_gpu_risk_kernel_production_E3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from imavis_edge_seg.config import ElasticityLevel
from imavis_edge_seg.router.calibrator import RiskCalibrator, fit_risk_calibrator
from imavis_edge_seg.router.policy import (
    _select_by_risk,
    _select_by_risk_and_latency_budget,
)
from imavis_edge_seg.router.risk_probe import compute_risk_score
from imavis_edge_seg.search.pareto import ParetoPoint

LEVELS: list[ElasticityLevel] = ["tiny", "small", "medium", "large"]
LATENCIES: dict[ElasticityLevel, float] = {"tiny": 1.0, "small": 2.0, "medium": 4.0, "large": 8.0}
ORDERED_POINTS = [
    ParetoPoint(level=lv, device_id="E3", backend="tensorrt_gpu", precision="fp16", latency_ms=LATENCIES[lv], miou=0.0, dataset="audit")
    for lv in LEVELS
]


def fit_synthetic_calibrator(rng: np.random.Generator, num_bins: int = 10) -> RiskCalibrator:
    """A real `fit_risk_calibrator` call (production code, not a reimplementation) on
    synthetic (raw_score, observed_error) pairs -- only the calibrator's own
    bin-quantization behavior matters for this decision-agreement check, not whether
    the synthetic errors are realistic."""
    raw_scores = rng.uniform(0, 3, size=500).tolist()
    observed_errors = rng.uniform(0, 1, size=500).tolist()
    return fit_risk_calibrator(raw_scores, observed_errors, num_bins=num_bins)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-outputs-dir", default="reports/real_outputs")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    calibrator_a = fit_synthetic_calibrator(rng)
    per_level_calibrators_d = {level: fit_synthetic_calibrator(rng) for level in LEVELS}

    numeric_results: dict[str, Any] = {}
    all_ref_scores: dict[str, list[float]] = {}
    all_gpu_scores: dict[str, list[float]] = {}

    for level in LEVELS:
        npz = np.load(f"{args.real_outputs_dir}/{level}.npz")
        logits = npz["logits"]  # (N, flat_size), float32, real TensorRT engine output
        gpu_scores = npz["gpu_risk_score"]  # (N,), computed on E3 via the real deployed path
        output_shape = tuple(int(x) for x in npz["output_shape"])  # (1, num_classes, H, W)
        _, num_classes, height, width = output_shape

        ref_scores = []
        for sample in logits:
            t = torch.from_numpy(sample.reshape(1, num_classes, height, width))
            # Real production call, no-ground-truth branch (matches real inference
            # deployment -- the same branch measure_router_overhead.py's numpy/gpu
            # backends both approximate).
            score = compute_risk_score(t, target=None)
            ref_scores.append(float(score.item()))

        ref_arr = np.array(ref_scores)
        abs_err = np.abs(ref_arr - gpu_scores)
        rel_err = abs_err / np.clip(np.abs(ref_arr), 1e-6, None)
        numeric_results[level] = {
            "n_samples": len(ref_scores),
            "ref_scores": ref_scores,
            "gpu_scores": gpu_scores.tolist(),
            "abs_error": {"mean": float(abs_err.mean()), "max": float(abs_err.max())},
            "rel_error": {"mean": float(rel_err.mean()), "max": float(rel_err.max())},
        }
        all_ref_scores[level] = ref_scores
        all_gpu_scores[level] = gpu_scores.tolist()
        print(f"{level}: n={len(ref_scores)} abs_err mean={abs_err.mean():.6e} max={abs_err.max():.6e}")

    # Decision agreement: treat "tiny" as the probe level (matches the real deployment
    # convention everywhere else in this project) -- for each of tiny's N real
    # samples, compare decision_a/decision_d computed from the real PyTorch reference
    # score vs. the real GPU-kernel score, using the REAL production policy functions.
    ref_tiny = all_ref_scores["tiny"]
    gpu_tiny = all_gpu_scores["tiny"]
    agree_a = agree_d = 0
    mismatches: list[dict[str, Any]] = []
    for ref_s, gpu_s in zip(ref_tiny, gpu_tiny, strict=True):
        # _select_by_risk takes an already-CALIBRATED risk (predict() applied), not
        # the raw entropy score -- matches real deployment (select_level's
        # calibrated_risk arg), calibrator_a/per_level_calibrators_d applied per raw
        # score below.
        da_ref = _select_by_risk(ORDERED_POINTS, calibrator_a.predict(ref_s), risk_target=0.1)
        da_gpu = _select_by_risk(ORDERED_POINTS, calibrator_a.predict(gpu_s), risk_target=0.1)
        per_level_risk_ref = {lv: per_level_calibrators_d[lv].predict(ref_s) for lv in LEVELS}
        per_level_risk_gpu = {lv: per_level_calibrators_d[lv].predict(gpu_s) for lv in LEVELS}
        dd_ref = _select_by_risk_and_latency_budget(ORDERED_POINTS, per_level_risk_ref, risk_target=0.1, latency_budget_ms=4.0)
        dd_gpu = _select_by_risk_and_latency_budget(ORDERED_POINTS, per_level_risk_gpu, risk_target=0.1, latency_budget_ms=4.0)
        if da_ref == da_gpu:
            agree_a += 1
        if dd_ref == dd_gpu:
            agree_d += 1
        if da_ref != da_gpu or dd_ref != dd_gpu:
            mismatches.append(
                {"ref_score": ref_s, "gpu_score": gpu_s, "decision_a": [da_ref, da_gpu], "decision_d": [dd_ref, dd_gpu]}
            )

    n = len(ref_tiny)
    decision_a_rate = agree_a / n
    decision_d_rate = agree_d / n
    decision_results: dict[str, Any] = {
        "n_trials": n,
        "decision_a_agreement_rate": decision_a_rate,
        "decision_d_agreement_rate": decision_d_rate,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
    }
    print(f"decision_a agreement: {decision_a_rate * 100:.3f}%")
    print(f"decision_d agreement: {decision_d_rate * 100:.3f}%")

    tolerance_abs_err = 1e-4
    tolerance_agreement_min = 0.999
    proposed_tolerance = {
        "scalar_risk_score_abs_error_nats": tolerance_abs_err,
        "decision_agreement_rate_min": tolerance_agreement_min,
    }
    max_abs_err = max(numeric_results[level]["abs_error"]["max"] for level in LEVELS)
    verdict = {
        "scalar_risk_score_abs_error_pass": max_abs_err <= tolerance_abs_err,
        "decision_a_pass": decision_a_rate >= tolerance_agreement_min,
        "decision_d_pass": decision_d_rate >= tolerance_agreement_min,
    }

    report = {
        "audit_type": "CUDA-kernel-versus-production-PyTorch-reference (real TensorRT engine outputs)",
        "reference_implementation": "imavis_edge_seg.router.risk_probe.compute_risk_score (real import)",
        "numeric_audit": numeric_results,
        "decision_agreement_audit": decision_results,
        "proposed_tolerance": proposed_tolerance,
        "verdict": verdict,
        "overall_pass": all(verdict.values()),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output_json}")
    print(f"OVERALL PASS: {report['overall_pass']}")


if __name__ == "__main__":
    main()
