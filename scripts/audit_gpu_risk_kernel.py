"""Correctness audit of the GPU-resident CUDA entropy kernel used by
`scripts/measure_router_overhead.py` (E3's overhead measurement) against the reference
softmax-entropy computation -- Codex's explicit, mandatory pre-manuscript-lock requirement
(`docs/COORDINATION_LOG.md` open thread #2, closing requirement 2 follow-up, 2026-09-28):
"E3 CUDA entropy/risk kernel phai duoc doi chieu voi PyTorch reference truoc manuscript
lock." This is a correctness audit, not a new experiment -- it isolates the kernel's own
numerics from any model-execution or backend difference by feeding BOTH implementations
the exact same synthetic logits array, never re-running inference.

Reference formula (verbatim from `src/imavis_edge_seg/router/risk_probe.py::
compute_risk_score`'s no-ground-truth branch, the real PyTorch implementation actually
used at inference time):
    probs = softmax(logits, dim=1)                    # channel axis
    entropy = -(probs * log(clamp_min(probs, 1e-12))).sum(dim=1)   # (H, W)
    risk = entropy.mean()

The numpy CPU implementation already duplicated in `measure_router_overhead.py` as
`compute_risk_score_from_logits` is a direct line-for-line transcription of this exact
formula (same axis, same clamp, same reduction order) -- used here as the numerical
reference. No torch install is needed on E3 for this: torch is never present on any edge
device in this project (only training servers), and the audit's purpose is to check the
CUDA kernel's arithmetic against a known-faithful transcription of the formula, not
against a second full model-execution backend (the earlier live smoke test already
confirmed the GPU-kernel path against a run of the actual model; that is E2E-relevant
but not what this isolates).

Usage:
    python3 audit_gpu_risk_kernel.py --output-json audit_gpu_risk_kernel_E3.json
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

if not hasattr(np, "bool"):
    setattr(np, "bool", bool)  # noqa: B010 -- TensorRT 8.5's __init__.py still uses the numpy<1.24 alias

import pycuda.autoinit  # noqa: F401  -- initializes the CUDA context
import pycuda.driver as cuda
import tensorrt as trt
from pycuda.compiler import SourceModule

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Verbatim copy of measure_router_overhead.py's kernel -- the exact kernel being audited,
# not a reimplementation (a copy that silently diverged from the real one would audit
# nothing).
_ENTROPY_KERNEL_SRC = """
extern "C" __global__ void entropy_kernel(const float *logits, float *entropy_out, int num_classes, int num_pixels) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= num_pixels) return;
    float max_val = -1e30f;
    for (int c = 0; c < num_classes; c++) {
        float v = logits[c * num_pixels + pixel];
        if (v > max_val) max_val = v;
    }
    float sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        sum_exp += expf(logits[c * num_pixels + pixel] - max_val);
    }
    float entropy = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        float p = expf(logits[c * num_pixels + pixel] - max_val) / sum_exp;
        entropy -= p * logf(fmaxf(p, 1e-12f));
    }
    entropy_out[pixel] = entropy;
}
"""
_entropy_module = SourceModule(_ENTROPY_KERNEL_SRC, no_extern_c=True)
_entropy_kernel = _entropy_module.get_function("entropy_kernel")

LEVELS = ["tiny", "small", "medium", "large"]


def get_output_shape(engine_path: str) -> tuple[int, int, int]:
    """Real (num_classes, H, W) output shape read directly from E3's actual, currently
    on-disk TensorRT engine file -- not a guessed/hardcoded shape, so this audit
    exercises the exact tensor sizes the kernel is deployed on."""
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            shape = tuple(engine.get_binding_shape(i))  # (1, num_classes, H, W)
            return (shape[1], shape[2], shape[3])
    raise RuntimeError(f"{engine_path}: no output binding found")


N_TRIALS_PER_LEVEL = 50
DECISION_TRIALS = 2000


def compute_risk_score_from_logits(flat_logits: np.ndarray, num_classes: int, num_pixels: int) -> np.ndarray:
    """Reference (numpy) path -- per-pixel entropy array, NOT yet mean-reduced (kept
    per-pixel here so the audit can compare the kernel's per-pixel output directly, a
    stricter check than comparing only the final scalar mean, which can hide localized
    disagreement)."""
    logits = flat_logits.reshape(num_classes, num_pixels)
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=0)
    return entropy.astype(np.float32)  # type: ignore[no-any-return]


def gpu_entropy(flat_logits: np.ndarray, num_classes: int, num_pixels: int) -> np.ndarray:
    device_in = cuda.mem_alloc(flat_logits.nbytes)
    device_out = cuda.mem_alloc(num_pixels * 4)
    cuda.memcpy_htod(device_in, flat_logits)
    block = 256
    grid = (num_pixels + block - 1) // block
    _entropy_kernel(device_in, device_out, np.int32(num_classes), np.int32(num_pixels), block=(block, 1, 1), grid=(grid, 1))
    cuda.Context.synchronize()
    host_out = np.empty(num_pixels, dtype=np.float32)
    cuda.memcpy_dtoh(host_out, device_out)
    return host_out


def fit_fake_calibrator(rng: np.random.Generator, num_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    raw_scores = rng.uniform(0, 3, size=500)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.quantile(raw_scores, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_expected_error = np.sort(rng.uniform(0, 1, size=num_bins))
    return bin_edges, bin_expected_error


def calibrator_predict(bin_edges: np.ndarray, bin_expected_error: np.ndarray, raw_score: float) -> float:
    bin_index = int(np.clip(np.searchsorted(bin_edges, raw_score, side="right") - 1, 0, len(bin_expected_error) - 1))
    return float(bin_expected_error[bin_index])


def decision_a(raw_score: float, calibrator: tuple[np.ndarray, np.ndarray], risk_target: float, latencies: dict[str, float]) -> str:
    ordered = sorted(latencies.items(), key=lambda kv: kv[1])
    risk = calibrator_predict(*calibrator, raw_score)
    if risk <= risk_target:
        return ordered[0][0]
    index = min(int(risk / risk_target), len(ordered) - 1)
    return ordered[index][0]


def decision_d(
    raw_score: float,
    per_level_calibrators: dict[str, tuple[np.ndarray, np.ndarray]],
    risk_target: float,
    latency_budget_ms: float,
    latencies: dict[str, float],
) -> str:
    ordered = sorted(latencies.items(), key=lambda kv: kv[1])
    per_level_risk = {level: calibrator_predict(*cal, raw_score) for level, cal in per_level_calibrators.items()}
    in_budget = [lv for lv, lat in ordered if lat <= latency_budget_ms]
    if not in_budget:
        return ordered[0][0]
    meeting = [lv for lv in in_budget if per_level_risk[lv] <= risk_target]
    if meeting:
        return min(meeting, key=lambda lv: latencies[lv])
    return min(in_budget, key=lambda lv: (per_level_risk[lv], latencies[lv]))


def run_numeric_audit(rng: np.random.Generator, level_shapes: dict[str, tuple[int, int, int]]) -> dict[str, Any]:
    """For each real output shape, N_TRIALS_PER_LEVEL random logits arrays (drawn from
    a distribution matching real untrained-network logit magnitudes, N(0, 3) -- wide
    enough to exercise the softmax's numerically-sensitive extremes, not just near-zero
    logits) -- both the per-pixel entropy array AND the final scalar mean are compared."""
    results: dict[str, Any] = {}
    for level, (num_classes, height, width) in level_shapes.items():
        num_pixels = height * width
        max_abs_errors = []
        mean_abs_errors = []
        max_rel_errors = []
        scalar_abs_errors = []
        for _ in range(N_TRIALS_PER_LEVEL):
            logits = rng.normal(0, 3, size=(num_classes, num_pixels)).astype(np.float32)
            flat = logits.ravel()
            ref = compute_risk_score_from_logits(flat, num_classes, num_pixels)
            gpu = gpu_entropy(flat, num_classes, num_pixels)
            abs_err = np.abs(ref - gpu)
            rel_err = abs_err / np.clip(np.abs(ref), 1e-6, None)
            max_abs_errors.append(float(abs_err.max()))
            mean_abs_errors.append(float(abs_err.mean()))
            max_rel_errors.append(float(rel_err.max()))
            scalar_abs_errors.append(float(abs(ref.mean() - gpu.mean())))
        results[level] = {
            "n_trials": N_TRIALS_PER_LEVEL,
            "per_pixel_max_abs_error": {"mean_over_trials": float(np.mean(max_abs_errors)), "max_over_trials": float(np.max(max_abs_errors))},
            "per_pixel_mean_abs_error": {"mean_over_trials": float(np.mean(mean_abs_errors)), "max_over_trials": float(np.max(mean_abs_errors))},
            "per_pixel_max_rel_error": {"mean_over_trials": float(np.mean(max_rel_errors)), "max_over_trials": float(np.max(max_rel_errors))},
            "scalar_risk_score_abs_error": {"mean_over_trials": float(np.mean(scalar_abs_errors)), "max_over_trials": float(np.max(scalar_abs_errors))},
        }
    return results


def run_decision_agreement_audit(rng: np.random.Generator, tiny_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Fit real (synthetic-data) calibrators as the overhead harness does, then draw
    DECISION_TRIALS pairs of (ref_score, gpu_score) from the SAME logits (using the
    tiny-level shape, decision cost is shape-independent) and check whether decision_a/
    decision_d agree between the two scores. A disagreement only arises when the two
    scores straddle a calibrator bin edge or a risk_target/latency_budget threshold --
    exactly the deployment-relevant question Codex asked ("route decisions giong nhau
    100%, ngoai tru tie nam trong tolerance da cong bo")."""
    num_classes, height, width = tiny_shape
    num_pixels = height * width
    calibrator_a = fit_fake_calibrator(rng)
    per_level_calibrators_d = {level: fit_fake_calibrator(rng) for level in LEVELS}
    latencies = {"tiny": 1.0, "small": 2.0, "medium": 4.0, "large": 8.0}
    risk_target = 0.1
    latency_budget_ms = 4.0

    agree_a = 0
    agree_d = 0
    mismatches: list[dict[str, Any]] = []
    for _ in range(DECISION_TRIALS):
        logits = rng.normal(0, 3, size=(num_classes, num_pixels)).astype(np.float32)
        flat = logits.ravel()
        ref_score = float(compute_risk_score_from_logits(flat, num_classes, num_pixels).mean())
        gpu_score = float(gpu_entropy(flat, num_classes, num_pixels).mean())

        da_ref = decision_a(ref_score, calibrator_a, risk_target, latencies)
        da_gpu = decision_a(gpu_score, calibrator_a, risk_target, latencies)
        dd_ref = decision_d(ref_score, per_level_calibrators_d, risk_target, latency_budget_ms, latencies)
        dd_gpu = decision_d(gpu_score, per_level_calibrators_d, risk_target, latency_budget_ms, latencies)

        if da_ref == da_gpu:
            agree_a += 1
        if dd_ref == dd_gpu:
            agree_d += 1
        if da_ref != da_gpu or dd_ref != dd_gpu:
            mismatches.append(
                {
                    "ref_score": ref_score,
                    "gpu_score": gpu_score,
                    "abs_diff": abs(ref_score - gpu_score),
                    "decision_a": [da_ref, da_gpu],
                    "decision_d": [dd_ref, dd_gpu],
                }
            )

    return {
        "n_trials": DECISION_TRIALS,
        "decision_a_agreement_rate": agree_a / DECISION_TRIALS,
        "decision_d_agreement_rate": agree_d / DECISION_TRIALS,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],  # cap for JSON size; count above is exact
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", default=".", help="directory containing pace_seg_{level}.engine (read-only, shapes only)")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    level_shapes: dict[str, tuple[int, int, int]] = {}
    for level in LEVELS:
        path = f"{args.engine_dir}/pace_seg_{level}.engine"
        level_shapes[level] = get_output_shape(path)
    print(f"real output shapes (num_classes, H, W) read from engines: {level_shapes}")

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    print("running numeric audit (per-pixel + scalar entropy, all 4 real output shapes)...")
    numeric = run_numeric_audit(rng, level_shapes)
    for level, stats in numeric.items():
        print(f"  {level}: max|abs err| over trials = {stats['per_pixel_max_abs_error']['max_over_trials']:.6e}, "
              f"scalar risk-score abs err (mean) = {stats['scalar_risk_score_abs_error']['mean_over_trials']:.6e}")

    print("running decision-agreement audit (A/D, 2000 trials)...")
    decisions = run_decision_agreement_audit(rng, level_shapes["tiny"])
    print(f"  decision_a agreement: {decisions['decision_a_agreement_rate'] * 100:.3f}%")
    print(f"  decision_d agreement: {decisions['decision_d_agreement_rate'] * 100:.3f}%")
    print(f"  n_mismatches: {decisions['n_mismatches']}/{decisions['n_trials']}")

    # Proposed tolerance (locked here, report actual observed numbers against it --
    # not tuned post-hoc): FP32 arithmetic over ~19-40 accumulated multiply-adds per
    # pixel plausibly accumulates ~1e-5 to 1e-4 absolute error; entropy's own range is
    # bounded by ln(num_classes) <~ 2.94 nats for 19 classes, so 1e-4 is a generous but
    # not vacuous bound relative to the signal's scale.
    proposed_tolerance = {
        "per_pixel_max_abs_error_nats": 1e-3,
        "scalar_risk_score_abs_error_nats": 1e-4,
        "decision_agreement_rate_min": 0.999,
    }
    all_max_abs = [numeric[level]["per_pixel_max_abs_error"]["max_over_trials"] for level in numeric]
    all_scalar_abs = [numeric[level]["scalar_risk_score_abs_error"]["max_over_trials"] for level in numeric]
    verdict = {
        "per_pixel_max_abs_error_pass": max(all_max_abs) <= proposed_tolerance["per_pixel_max_abs_error_nats"],
        "scalar_risk_score_abs_error_pass": max(all_scalar_abs) <= proposed_tolerance["scalar_risk_score_abs_error_nats"],
        "decision_a_pass": decisions["decision_a_agreement_rate"] >= proposed_tolerance["decision_agreement_rate_min"],
        "decision_d_pass": decisions["decision_d_agreement_rate"] >= proposed_tolerance["decision_agreement_rate_min"],
    }

    report = {
        "elapsed_s": time.time() - t0,
        "level_shapes": {k: list(v) for k, v in level_shapes.items()},
        "n_trials_per_level": N_TRIALS_PER_LEVEL,
        "decision_trials": DECISION_TRIALS,
        "numeric_audit": numeric,
        "decision_agreement_audit": decisions,
        "proposed_tolerance": proposed_tolerance,
        "verdict": verdict,
        "overall_pass": all(verdict.values()),
    }
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output_json}")
    print(f"OVERALL PASS: {report['overall_pass']}")


if __name__ == "__main__":
    main()
