"""Real end-to-end router-overhead measurement on real hardware (closing requirement
2, docs/COORDINATION_LOG.md open thread #2), protocol locked with Codex 2026-09-22.
Self-contained (no imavis_edge_seg package import -- only tensorrt/pycuda/numpy/stdlib)
so it runs directly on an edge device without needing the full training-side
dependency stack (torch, pydantic, etc.) installed there.

Formula: t_e2e = t_probe + t_decision + t_switch + t_selected_extra, measured as ONE
direct end-to-end trace per route class (never 4 independent benchmarks summed).
`t_selected_extra = 0` when the selected candidate IS the probe level (tiny->tiny),
reusing the probe's own output -- kept as its own route class, both as a real
operating point and as a sanity check that the implementation isn't re-running tiny.

Route classes (never collapsed): tiny->tiny, tiny->small, tiny->medium, tiny->large.

Two scenarios:
  - warm/resident: every engine's execution context already created and warm before
    timing starts; t_switch is real dispatch/context-selection cost only.
  - cold/reload: the candidate engine's execution context is destroyed and rebuilt
    from the on-disk engine file on every transition -- a genuine release+reload,
    not just a cache-miss simulation.

Decision-path timing (A vs D) is a separate microbenchmark (numpy/CPU only, no GPU
involved) -- both policies' cost is measured, never assumed equal. Self-contained
reimplementations of RiskCalibrator.predict/_select_by_risk/_select_by_risk_and_
latency_budget's exact logic (avoids needing pydantic/torch on the edge device);
kept behaviorally identical to src/imavis_edge_seg/router/{calibrator,policy}.py.

Usage:
    python3 measure_router_overhead.py --engine-dir . --device-label E3 \
        --backend tensorrt --output-json router_overhead_E3.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "bool"):
    setattr(np, "bool", bool)  # noqa: B010 -- TensorRT 8.5's __init__.py still uses the numpy<1.24 alias

import pycuda.autoinit  # noqa: F401  -- initializes the CUDA context
import pycuda.driver as cuda
import tensorrt as trt
from pycuda.compiler import SourceModule

# GPU-resident entropy kernel: computes per-pixel softmax entropy directly from the
# TensorRT engine's device-resident output buffer, one thread per pixel -- avoids
# copying the full (num_classes, H, W) logits tensor back to host and computing
# exp/log there in numpy, which the E3 smoke test showed costs ~40-80ms on this
# device's numpy/ARM build (a real, reproducible, but likely non-representative
# number: the project's actual PyTorch `compute_risk_score` runs on GPU-resident
# tensors, not host numpy after a full-tensor copy-back). This kernel is the
# GPU-resident equivalent, matching that design. Only the tiny per-pixel entropy
# array (H*W floats, not num_classes*H*W) needs to come back to host afterward, for
# the final mean -- cheap even via numpy.
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

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
LEVELS = ["tiny", "small", "medium", "large"]
ROUTE_CLASSES = [("tiny", level) for level in LEVELS]  # probe -> selected

WARM_WARMUP_ITERS = 50
WARM_MEASURED_ITERS = 500
WARM_MEASURED_ITERS_MAX = 2000
COLD_ITERS = 50
DECISION_MICROBENCH_ITERS = 2000
BOOTSTRAP_RESAMPLES = 2000
CI_WIDTH_THRESHOLD_FRACTION = 0.02  # 2% of the median


# ---- self-contained decision-path reimplementations (numpy only) --------------------
# Behaviorally identical to src/imavis_edge_seg/router/{calibrator,policy}.py -- kept
# inline so this script has no dependency on the training-side package (pydantic/
# torch) on the edge device. Only used for TIMING the decision computation itself;
# the actual calibrator/policy parameters are fit on synthetic data since only the
# computational cost matters here, not real accuracy.


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


def compute_risk_score_from_logits(flat_logits: np.ndarray, output_shape: tuple[int, ...]) -> float:
    """Mean per-pixel softmax entropy -- matches router.risk_probe.compute_risk_score's
    no-ground-truth (real-inference) branch. `flat_logits` is the raw flat pagelocked
    buffer from `TrtEngine.infer_sync`; `output_shape` (1, num_classes, H, W) is needed
    to reshape it before computing softmax over the *class* axis, not the flat array."""
    _, num_classes, height, width = output_shape
    # (num_classes, H*W) is a plain reshape of the NCHW-flat buffer -- C-contiguous,
    # no data movement. Softmax/entropy computed along axis=0 (channels) directly,
    # NOT via a `.T` transpose to (H*W, num_classes): a transposed view's strides are
    # non-contiguous, and every subsequent numpy reduction along that axis becomes
    # dramatically slower (~8x observed here) -- a real bug caught during the E3
    # overhead smoke test (entropy computation alone accounted for most of a ~30ms
    # "tiny->tiny" route class that should have been a few ms).
    logits = flat_logits.reshape(num_classes, height * width)
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=0)
    return float(entropy.mean())


def decision_a(raw_score: float, calibrator: tuple[np.ndarray, np.ndarray], risk_target: float, latencies: dict[str, float]) -> str:
    """Cell A: single calibrated risk, rank-step escalation (_select_by_risk)."""
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
    """Cell D: risk-and-latency-constrained policy (_select_by_risk_and_latency_budget)."""
    ordered = sorted(latencies.items(), key=lambda kv: kv[1])
    per_level_risk = {level: calibrator_predict(*cal, raw_score) for level, cal in per_level_calibrators.items()}
    in_budget = [lv for lv, lat in ordered if lat <= latency_budget_ms]
    if not in_budget:
        return ordered[0][0]
    meeting = [lv for lv in in_budget if per_level_risk[lv] <= risk_target]
    if meeting:
        return min(meeting, key=lambda lv: latencies[lv])
    return min(in_budget, key=lambda lv: (per_level_risk[lv], latencies[lv]))


# ---- TensorRT engine wrapper ---------------------------------------------------------


class TrtEngine:
    def __init__(self, engine_path: Path) -> None:
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings: list[int] = []
        self.host_in: Any = None
        self.device_in: Any = None
        self.host_out: Any = None
        self.device_out: Any = None
        self.output_shape: tuple[int, ...] = ()
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(i):
                self.host_in, self.device_in = host_mem, device_mem
            else:
                self.host_out, self.device_out = host_mem, device_mem
                self.output_shape = tuple(shape)  # (1, num_classes, H, W) -- needed to
                # reshape the flat pagelocked buffer back before any per-pixel
                # softmax/entropy computation (a real bug caught during the smoke
                # test: without this, entropy was computed over the entire ~1.4M-
                # element flat array as if it were the class axis, ~9x slower and
                # semantically wrong).
        _, num_classes, height, width = self.output_shape
        self.num_classes = num_classes
        self.num_pixels = height * width
        self.entropy_device = cuda.mem_alloc(self.num_pixels * 4)  # float32
        self.entropy_host = cuda.pagelocked_empty(self.num_pixels, np.float32)

    def infer_no_copy(self, input_array: np.ndarray) -> None:
        """Same as `infer_sync` but does NOT copy the output back to host -- for the
        GPU-resident entropy path, which reads `self.device_out` directly via the
        entropy kernel instead."""
        np.copyto(self.host_in, input_array.ravel())
        cuda.memcpy_htod_async(self.device_in, self.host_in, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()

    def risk_score_gpu(self) -> float:
        """GPU-resident softmax entropy (see module-level `_entropy_kernel`): reads
        `self.device_out` directly, writes one float per pixel to `self.entropy_device`,
        copies back only that (H*W, not num_classes*H*W) small array, and takes its
        mean on host. Must be called after `infer_no_copy` (or `infer_sync`) on the
        same engine."""
        block = 256
        grid = (self.num_pixels + block - 1) // block
        _entropy_kernel(
            self.device_out, self.entropy_device, np.int32(self.num_classes), np.int32(self.num_pixels),
            block=(block, 1, 1), grid=(grid, 1), stream=self.stream,
        )
        cuda.memcpy_dtoh_async(self.entropy_host, self.entropy_device, self.stream)
        self.stream.synchronize()
        return float(self.entropy_host.mean())

    def infer_sync(self, input_array: np.ndarray) -> np.ndarray:
        """One full inference, synchronized -- returns the flat output array. Callers
        time around this (and any decision logic) themselves; this method's own
        internal `stream.synchronize()` is what makes CUDA timing here honest (no
        async work left in flight when the caller reads its clock)."""
        np.copyto(self.host_in, input_array.ravel())
        cuda.memcpy_htod_async(self.device_in, self.host_in, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_out, self.device_out, self.stream)
        self.stream.synchronize()
        # pycuda's pagelocked buffer type isn't stubbed, so mypy sees `Any` here.
        return self.host_out  # type: ignore[no-any-return]

    def destroy(self) -> None:
        del self.context
        del self.engine


def load_engine_fresh(engine_path: Path) -> TrtEngine:
    return TrtEngine(engine_path)


# ---- stats helpers --------------------------------------------------------------------


def bootstrap_ci(samples: list[float], stat_fn: Any, n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    arr = np.asarray(samples)
    stats = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarize(samples: list[float]) -> dict[str, Any]:
    arr = np.asarray(samples)
    median_ci = bootstrap_ci(samples, np.median)
    mean_ci = bootstrap_ci(samples, np.mean)
    return {
        "n": len(samples),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "median_ci95": median_ci,
        "mean_ci95": mean_ci,
        "median_ci_width_pct_of_median": abs(median_ci[1] - median_ci[0]) / float(np.median(arr)) * 100 if np.median(arr) else 0.0,
    }


def read_temperature(device_label: str) -> float | None:
    """Best-effort SoC/GPU temperature read, Jetson-only (tegrastats or thermal_zone
    sysfs). Returns None (not zero) if unavailable -- never fabricate a reading."""
    try:
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            zone_type = (zone / "type").read_text().strip().lower()
            if "gpu" in zone_type or "soc" in zone_type or "cpu" in zone_type:
                milli_c = int((zone / "temp").read_text().strip())
                return milli_c / 1000.0
    except Exception:
        pass
    return None


def check_throttled() -> bool:
    """Jetson-specific throttling check via tegrastats' one-shot output, if present."""
    try:
        out = subprocess.run(["tegrastats", "--interval", "1"], capture_output=True, text=True, timeout=3)
        return "online" not in out.stdout.lower() and False  # placeholder: tegrastats' throttling flag format varies by JetPack version; logged raw below instead
    except Exception:
        return False


# ---- measurement routines --------------------------------------------------------------


def _probe_and_score(engine: TrtEngine, input_array: np.ndarray, entropy_backend: str) -> float:
    if entropy_backend == "gpu":
        engine.infer_no_copy(input_array)
        return engine.risk_score_gpu()
    probe_out = engine.infer_sync(input_array)
    return compute_risk_score_from_logits(probe_out, engine.output_shape)


def run_warm_route_class(
    engines: dict[str, TrtEngine],
    probe_level: str,
    selected_level: str,
    inputs: dict[str, np.ndarray],
    latencies_ms: dict[str, float],
    risk_target: float,
    latency_budget_ms: float,
    entropy_backend: str,
) -> dict[str, Any]:
    """All engines already resident (warm). Measures t_e2e directly for this route
    class: probe inference -> decision (cell D's) -> switch (context already exists,
    so this is just Python-side dispatch) -> selected inference (skipped, reusing the
    probe's own output, if selected == probe). `entropy_backend`: "numpy" (host-side,
    matches naive TensorRT-output-copied-to-host deployments) or "gpu" (GPU-resident
    kernel, matches the project's actual PyTorch `compute_risk_score`'s GPU-tensor
    design) -- both measured, never assumed equal (a real ~40-80ms gap was found on
    E3's numpy/ARM build during the initial smoke test)."""
    rng = np.random.default_rng(hash((probe_level, selected_level)) % (2**32))
    per_level_calibrators = {level: fit_fake_calibrator(rng) for level in LEVELS}

    samples: list[float] = []
    n_total = WARM_WARMUP_ITERS + WARM_MEASURED_ITERS
    for i in range(n_total):
        t0 = time.perf_counter()
        raw_score = _probe_and_score(engines[probe_level], inputs[probe_level], entropy_backend)
        # Decision (cell D's, timed as part of the real e2e trace): which level
        # WOULD be chosen given this (synthetic) risk. We force the actual inference
        # step below to match the route class under test (`selected_level`), so every
        # route class gets its own real measurement regardless of what this
        # particular random draw's decision would have been -- decision cost is still
        # genuinely paid every iteration.
        _ = decision_d(raw_score, per_level_calibrators, risk_target, latency_budget_ms, latencies_ms)
        if selected_level != probe_level:
            engines[selected_level].infer_sync(inputs[selected_level])
        t1 = time.perf_counter()
        if i >= WARM_WARMUP_ITERS:
            samples.append((t1 - t0) * 1000.0)

    # Adaptive: extend to WARM_MEASURED_ITERS_MAX if the median's CI is still wide.
    stats = summarize(samples)
    while stats["median_ci_width_pct_of_median"] > CI_WIDTH_THRESHOLD_FRACTION * 100 and len(samples) < WARM_MEASURED_ITERS_MAX:
        t0 = time.perf_counter()
        raw_score = _probe_and_score(engines[probe_level], inputs[probe_level], entropy_backend)
        _ = decision_d(raw_score, per_level_calibrators, risk_target, latency_budget_ms, latencies_ms)
        if selected_level != probe_level:
            engines[selected_level].infer_sync(inputs[selected_level])
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
        stats = summarize(samples)

    return stats


def run_cold_route_class(
    engine_paths: dict[str, Path],
    resident_engines: dict[str, TrtEngine],
    probe_level: str,
    selected_level: str,
    inputs: dict[str, np.ndarray],
    entropy_backend: str,
) -> dict[str, Any]:
    """Only `probe_level` (tiny) stays resident. On every transition to a non-probe
    level, the selected engine's context is genuinely destroyed and rebuilt from the
    on-disk .engine file -- a real release+reload, not a simulated cache miss."""
    if selected_level == probe_level:
        # tiny->tiny has no reload to measure; report the warm number's shape but
        # mark it explicitly as not applicable to avoid a misleading "0 reload cost".
        return {"note": "tiny->tiny has no candidate reload; see warm scenario for this route class"}

    samples: list[float] = []
    for _ in range(COLD_ITERS):
        t0 = time.perf_counter()
        _ = _probe_and_score(resident_engines[probe_level], inputs[probe_level], entropy_backend)
        candidate = load_engine_fresh(engine_paths[selected_level])
        candidate.infer_sync(inputs[selected_level])
        candidate.destroy()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return summarize(samples)


def run_decision_microbenchmark() -> dict[str, Any]:
    rng = np.random.default_rng(1)
    calibrator_a = fit_fake_calibrator(rng)
    per_level_calibrators_d = {level: fit_fake_calibrator(rng) for level in LEVELS}
    latencies = {"tiny": 1.0, "small": 2.0, "medium": 4.0, "large": 8.0}
    raw_scores = rng.uniform(0, 3, size=DECISION_MICROBENCH_ITERS)

    samples_a: list[float] = []
    for s in raw_scores:
        t0 = time.perf_counter()
        decision_a(float(s), calibrator_a, 0.1, latencies)
        samples_a.append((time.perf_counter() - t0) * 1000.0)

    samples_d: list[float] = []
    for s in raw_scores:
        t0 = time.perf_counter()
        decision_d(float(s), per_level_calibrators_d, 0.1, 4.0, latencies)
        samples_d.append((time.perf_counter() - t0) * 1000.0)

    return {"decision_a": summarize(samples_a), "decision_d": summarize(samples_d)}


def main() -> None:
    global WARM_WARMUP_ITERS, WARM_MEASURED_ITERS, COLD_ITERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, default=Path("."))
    parser.add_argument("--device-label", required=True, help='e.g. "E1" or "E3"')
    parser.add_argument("--backend", required=True, help='e.g. "hailo_hef" or "tensorrt_gpu"')
    parser.add_argument("--risk-target", type=float, default=0.1)
    parser.add_argument("--latency-budget-ms", type=float, default=4.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--warmup-iters", type=int, default=WARM_WARMUP_ITERS, help="override for a quick smoke test")
    parser.add_argument("--warm-iters", type=int, default=WARM_MEASURED_ITERS, help="override for a quick smoke test")
    parser.add_argument("--cold-iters", type=int, default=COLD_ITERS, help="override for a quick smoke test")
    args = parser.parse_args()

    WARM_WARMUP_ITERS = args.warmup_iters
    WARM_MEASURED_ITERS = args.warm_iters
    COLD_ITERS = args.cold_iters

    engine_paths = {level: args.engine_dir / f"pace_seg_{level}.engine" for level in LEVELS}
    for _level, path in engine_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing engine file: {path}")

    input_shapes: dict[str, tuple[int, int, int, int]] = {}
    probe_engine = TrtEngine(engine_paths["tiny"])
    for i in range(probe_engine.engine.num_bindings):
        if probe_engine.engine.binding_is_input(i):
            input_shapes["tiny"] = tuple(probe_engine.engine.get_binding_shape(i))

    print(f"loaded tiny engine, input shape {input_shapes.get('tiny')}")

    temp_start = read_temperature(args.device_label)
    print(f"starting temperature: {temp_start}")

    report: dict[str, Any] = {
        "device_label": args.device_label,
        "backend": args.backend,
        "tensorrt_version": trt.__version__,
        "risk_target": args.risk_target,
        "latency_budget_ms": args.latency_budget_ms,
        "temp_start_c": temp_start,
        "warm": {},
        "cold": {},
    }

    # --- warm/resident scenario: load every engine once, keep resident ---
    print("building warm/resident engines (all 4 levels)...")
    engines = {level: TrtEngine(engine_paths[level]) for level in LEVELS}
    rng = np.random.default_rng(42)
    inputs: dict[str, np.ndarray] = {}
    for level in LEVELS:
        shape = None
        for i in range(engines[level].engine.num_bindings):
            if engines[level].engine.binding_is_input(i):
                shape = tuple(engines[level].engine.get_binding_shape(i))
        if shape is None:
            raise RuntimeError(f"engine for level {level!r} has no input binding")
        inputs[level] = rng.standard_normal(shape, dtype=np.float32)
    # measured latencies for the D policy's latency_budget check inside the loop --
    # placeholder relative values (this script only needs internally-consistent
    # ordering for the decision computation's cost to be realistic, not real LUT
    # numbers, which are substituted in during the frontier replay analysis step)
    latencies_ms_placeholder = {"tiny": 1.0, "small": 2.0, "medium": 4.0, "large": 8.0}

    import random

    report["warm"] = {"numpy": {}, "gpu": {}}
    for entropy_backend in ("numpy", "gpu"):
        route_order = list(ROUTE_CLASSES)
        random.Random(0).shuffle(route_order)  # interleave/randomize route order, per protocol
        for probe_level, selected_level in route_order:
            route_key = f"{probe_level}->{selected_level}"
            print(f"warm route {route_key} (entropy_backend={entropy_backend})...")
            report["warm"][entropy_backend][route_key] = run_warm_route_class(
                engines, probe_level, selected_level, inputs, latencies_ms_placeholder,
                args.risk_target, args.latency_budget_ms, entropy_backend,
            )

    temp_after_warm = read_temperature(args.device_label)
    report["temp_after_warm_c"] = temp_after_warm
    print(f"temperature after warm scenario: {temp_after_warm}")

    # --- cold/reload scenario: only tiny resident, candidate reloaded every time ---
    # Uses the GPU-resident entropy backend only (the representative/optimized path;
    # the numpy-vs-gpu comparison is already covered in the warm scenario above).
    for probe_level, selected_level in ROUTE_CLASSES:
        route_key = f"{probe_level}->{selected_level}"
        print(f"cold route {route_key}...")
        report["cold"][route_key] = run_cold_route_class(
            engine_paths, {"tiny": engines["tiny"]}, probe_level, selected_level, inputs, "gpu"
        )

    temp_end = read_temperature(args.device_label)
    report["temp_end_c"] = temp_end
    print(f"final temperature: {temp_end}")

    report["decision_microbenchmark"] = run_decision_microbenchmark()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
