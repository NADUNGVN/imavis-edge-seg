"""Real end-to-end router-overhead measurement on E1 (Hailo-8, Pi5) -- the second
mandatory backend for closing requirement 2 (docs/COORDINATION_LOG.md open thread #2),
protocol locked with Codex 2026-09-22. Mirrors scripts/measure_router_overhead.py's
TensorRT/pycuda harness (used on E3) as closely as the hardware allows; differences from
that script are architectural, not shortcuts, and are called out inline.

Formula: t_e2e = t_probe + t_decision + t_switch + t_selected_extra, measured as ONE
direct end-to-end trace per route class (never summed from independent benchmarks).
`t_selected_extra = 0` when the selected candidate IS the probe level (tiny->tiny),
reusing the probe's own output.

Route classes (never collapsed): tiny->tiny, tiny->small, tiny->medium, tiny->large.

Architectural difference from E3 (disclose, do not paper over):
  - Hailo-8 (this chip, classic single-context HEF, no scheduler) can only have ONE
    network group hardware-activated at a time -- unlike TensorRT/CUDA, where multiple
    execution contexts coexist and "switch" is just Python-side dispatch. So even in the
    warm/resident scenario, every route class that leaves the probe level pays a real
    activate(probe)/infer/deactivate -> activate(selected)/infer/deactivate cycle. This
    activate/deactivate cost IS t_switch here, not an artifact -- confirmed live (see
    _hailo_smoke_explore2.py): inferring on a non-activated network group raises
    HailoRTNetworkGroupNotActivatedException, and switching between two groups via
    sequential (non-nested) `with group.activate(params):` blocks works cleanly.
  - No GPU-resident entropy kernel is possible on Hailo's dataflow architecture (no
    general-purpose compute-shader model like CUDA) -- only the numpy/host backend
    exists here. `report["warm"]["gpu"]` is intentionally omitted (not zero, not
    approximated) -- see the "Not yet" note this script prints at the end.
  - Output tensors are NHWC (channels-last), not NCHW -- softmax/entropy reduces over
    the LAST axis, not axis 0.

Usage:
    python3 measure_router_overhead_hailo.py --hef-dir . --device-label E1 \
        --backend hailo_hef --output-json router_overhead_E1.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import hailo_platform as hp
import numpy as np

LEVELS = ["tiny", "small", "medium", "large"]
ROUTE_CLASSES = [("tiny", level) for level in LEVELS]

WARM_WARMUP_ITERS = 50
WARM_MEASURED_ITERS = 500
WARM_MEASURED_ITERS_MAX = 2000
COLD_ITERS = 50
DECISION_MICROBENCH_ITERS = 2000
BOOTSTRAP_RESAMPLES = 2000
CI_WIDTH_THRESHOLD_FRACTION = 0.02  # 2% of the median


# ---- self-contained decision-path reimplementations (numpy only) --------------------
# Identical to scripts/measure_router_overhead.py's copies (hardware-agnostic, pure
# numpy/CPU) -- kept duplicated rather than imported so this script stays self-contained
# on the edge device (no imavis_edge_seg package, no shared-module path dependency).


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


def compute_risk_score_from_logits_hwc(output_array: np.ndarray) -> float:
    """Mean per-pixel softmax entropy -- matches router.risk_probe.compute_risk_score's
    no-ground-truth branch. `output_array` is (1, H, W, num_classes) or (H, W, num_classes)
    -- Hailo vstreams are NHWC (channels-last), so softmax/entropy reduces over the LAST
    axis, the opposite of E3's NCHW TensorRT buffers (axis 0)."""
    arr = output_array.reshape(-1, output_array.shape[-1])  # (H*W, num_classes), C-contiguous
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1)
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


# ---- HailoRT engine wrapper -----------------------------------------------------------


class HailoEngine:
    """One HEF configured onto a shared VDevice. Configuring is cheap and done once per
    engine at startup (warm/resident); each `infer_sync` call still pays a real
    activate()/deactivate() cycle -- see module docstring's architectural-difference
    note. `ng_params` created once and reused (matches HailoRT examples)."""

    def __init__(self, vdevice: Any, hef_path: Path) -> None:
        self.hef = hp.HEF(str(hef_path))
        configure_params = hp.ConfigureParams.create_from_hef(self.hef, interface=hp.HailoStreamInterface.PCIe)
        self.network_group = vdevice.configure(self.hef, configure_params)[0]
        self.ng_params = self.network_group.create_params()
        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_info = self.hef.get_output_vstream_infos()[0]
        input_vstreams_params = hp.InputVStreamParams.make(self.network_group, format_type=hp.FormatType.FLOAT32)
        output_vstreams_params = hp.OutputVStreamParams.make(self.network_group, format_type=hp.FormatType.FLOAT32)
        self._pipeline_cm = hp.InferVStreams(self.network_group, input_vstreams_params, output_vstreams_params)
        self.pipeline = self._pipeline_cm.__enter__()
        self.input_shape: tuple[int, ...] = tuple(self.in_info.shape)
        self.output_shape: tuple[int, ...] = tuple(self.out_info.shape)

    def infer_sync(self, input_array: np.ndarray) -> np.ndarray:
        """One full inference: activate this network group, run, deactivate. The
        activate/deactivate pair is real hardware cost on this single-context chip, not
        Python-side bookkeeping -- included deliberately, timed by the caller."""
        with self.network_group.activate(self.ng_params):
            out = self.pipeline.infer({self.in_info.name: input_array})
        return out[self.out_info.name]  # type: ignore[no-any-return]

    def close(self) -> None:
        """Exit the InferVStreams context, then drop every reference to the
        network group/HEF and force garbage collection. HailoRT ties the
        configured core-op's release to the C-extension object's destructor, not
        to any explicit `.release()`/`.deconfigure()` call (none exists on
        `ConfiguredNetwork` -- checked via `dir()`) -- a real bug caught here:
        without the explicit `del` + `gc.collect()`, the cold-reload loop's
        repeated `vdevice.configure()` calls exhausted this chip's 32-core-op
        budget partway through a single route class, since the previous
        iteration's network group was still alive (pending GC) when the next
        one was configured."""
        self._pipeline_cm.__exit__(None, None, None)
        del self.pipeline
        del self.network_group
        del self.hef
        gc.collect()


# ---- stats helpers ----------------------------------------------------------------------


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


def read_temperature() -> float | None:
    """Best-effort SoC temperature read via thermal_zone sysfs (Pi5-generic). Returns
    None (not zero) if unavailable -- never fabricate a reading."""
    try:
        best = None
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            zone_type = (zone / "type").read_text().strip().lower()
            if "cpu" in zone_type or "soc" in zone_type:
                milli_c = int((zone / "temp").read_text().strip())
                best = milli_c / 1000.0
                break
        return best
    except Exception:
        return None


def read_hailo_temperature(device_handle: Any) -> float | None:
    """Best-effort Hailo-8 chip temperature via HailoRT's own control call (on-chip
    telemetry, same caveat as the project's existing E1 benchmark protocol note: this is
    not an external calibrated meter)."""
    try:
        info = device_handle.control.get_chip_temperature()
        return float(info.ts0_temperature)
    except Exception:
        return None


# ---- measurement routines --------------------------------------------------------------


def run_warm_route_class(
    engines: dict[str, HailoEngine],
    probe_level: str,
    selected_level: str,
    inputs: dict[str, np.ndarray],
    latencies_ms: dict[str, float],
    risk_target: float,
    latency_budget_ms: float,
) -> dict[str, Any]:
    """All 4 network groups already configured (warm/resident). Measures t_e2e directly:
    probe inference (with its real activate/deactivate cost) -> decision (cell D's) ->
    switch+selected inference (skipped, reusing the probe's own output, if
    selected == probe)."""
    rng = np.random.default_rng(hash((probe_level, selected_level)) % (2**32))
    per_level_calibrators = {level: fit_fake_calibrator(rng) for level in LEVELS}

    samples: list[float] = []
    n_total = WARM_WARMUP_ITERS + WARM_MEASURED_ITERS
    for i in range(n_total):
        t0 = time.perf_counter()
        probe_out = engines[probe_level].infer_sync(inputs[probe_level])
        raw_score = compute_risk_score_from_logits_hwc(probe_out)
        _ = decision_d(raw_score, per_level_calibrators, risk_target, latency_budget_ms, latencies_ms)
        if selected_level != probe_level:
            engines[selected_level].infer_sync(inputs[selected_level])
        t1 = time.perf_counter()
        if i >= WARM_WARMUP_ITERS:
            samples.append((t1 - t0) * 1000.0)

    stats = summarize(samples)
    while stats["median_ci_width_pct_of_median"] > CI_WIDTH_THRESHOLD_FRACTION * 100 and len(samples) < WARM_MEASURED_ITERS_MAX:
        t0 = time.perf_counter()
        probe_out = engines[probe_level].infer_sync(inputs[probe_level])
        raw_score = compute_risk_score_from_logits_hwc(probe_out)
        _ = decision_d(raw_score, per_level_calibrators, risk_target, latency_budget_ms, latencies_ms)
        if selected_level != probe_level:
            engines[selected_level].infer_sync(inputs[selected_level])
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
        stats = summarize(samples)

    return stats


def run_cold_route_class(
    hef_paths: dict[str, Path],
    probe_level: str,
    selected_level: str,
    inputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Real, hardware-forced design difference from E3's cold scenario (disclose, do
    not paper over): this Hailo-8 M.2 module allows exactly ONE VDevice at a time --
    confirmed live (see _hailo_smoke_explore3.py): opening a second VDevice while the
    first is still open raises HAILO_OUT_OF_PHYSICAL_DEVICES, and a VDevice's
    configured network groups can never be individually released (no such method
    exists; `del` + `gc.collect()` does not free the core-op slot either -- confirmed
    live, `HAILO_INVALID_OPERATION` after ~28 accumulated reconfigures against this
    chip's 32-core-op ceiling). So unlike E3 (candidate-only reload, probe stays
    resident), Hailo's cold scenario reconfigures the ENTIRE VDevice fresh every
    iteration -- probe included, even for tiny->tiny. This is a stricter (more
    pessimistic) cold-reload number than E3's, not a shortcut."""
    samples: list[float] = []
    for _ in range(COLD_ITERS):
        t0 = time.perf_counter()
        with hp.VDevice() as vdevice:
            probe = HailoEngine(vdevice, hef_paths[probe_level])
            probe_out = probe.infer_sync(inputs[probe_level])
            _ = compute_risk_score_from_logits_hwc(probe_out)
            if selected_level != probe_level:
                candidate = HailoEngine(vdevice, hef_paths[selected_level])
                candidate.infer_sync(inputs[selected_level])
                candidate.close()
            probe.close()
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
    parser.add_argument("--hef-dir", type=Path, default=Path("."))
    parser.add_argument("--device-label", required=True, help='e.g. "E1"')
    parser.add_argument("--backend", required=True, help='e.g. "hailo_hef"')
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

    hef_paths = {level: args.hef_dir / f"pace_seg_{level}.hef" for level in LEVELS}
    for _level, path in hef_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing HEF file: {path}")

    temp_start = read_temperature()
    print(f"starting CPU/SoC temperature: {temp_start}")

    report: dict[str, Any] = {
        "device_label": args.device_label,
        "backend": args.backend,
        "entropy_backend_available": ["numpy"],
        "entropy_backend_note": (
            "Hailo-8's dataflow architecture has no general-purpose compute-shader "
            "model (unlike CUDA on E3) -- no GPU-resident entropy kernel is possible "
            "here. Only the numpy/host backend is measured; report only against E3's "
            "numpy backend when comparing, not E3's gpu backend."
        ),
        "risk_target": args.risk_target,
        "latency_budget_ms": args.latency_budget_ms,
        "temp_start_c": temp_start,
        "warm": {},
        "cold": {},
    }

    with hp.VDevice() as vdevice:
        report["hailo_temp_start_c"] = read_hailo_temperature(vdevice.get_physical_devices()[0])

        print("configuring warm/resident engines (all 4 levels)...")
        engines = {level: HailoEngine(vdevice, hef_paths[level]) for level in LEVELS}
        rng = np.random.default_rng(42)
        inputs: dict[str, np.ndarray] = {}
        for level in LEVELS:
            shape = engines[level].input_shape
            inputs[level] = rng.standard_normal((1, *shape), dtype=np.float32)
        input_shapes = {level: engines[level].input_shape for level in LEVELS}
        print(f"input shapes (H, W, C): {input_shapes}")

        latencies_ms_placeholder = {"tiny": 1.0, "small": 2.0, "medium": 4.0, "large": 8.0}

        report["warm"] = {"numpy": {}}
        route_order = list(ROUTE_CLASSES)
        random.Random(0).shuffle(route_order)  # interleave/randomize route order, per protocol
        for probe_level, selected_level in route_order:
            route_key = f"{probe_level}->{selected_level}"
            print(f"warm route {route_key}...")
            report["warm"]["numpy"][route_key] = run_warm_route_class(
                engines, probe_level, selected_level, inputs, latencies_ms_placeholder,
                args.risk_target, args.latency_budget_ms,
            )

        temp_after_warm = read_temperature()
        report["temp_after_warm_c"] = temp_after_warm
        report["hailo_temp_after_warm_c"] = read_hailo_temperature(vdevice.get_physical_devices()[0])
        print(f"temperature after warm scenario: {temp_after_warm}")

        for level in LEVELS:
            engines[level].close()
    # `vdevice` (warm scenario) fully closed above -- this chip allows exactly one
    # VDevice at a time (confirmed live: a second concurrent VDevice raises
    # HAILO_OUT_OF_PHYSICAL_DEVICES), so the cold scenario below cannot start until
    # every warm-scenario handle is released.

    # --- cold/reload scenario: entire VDevice (probe included) reconfigured fresh
    # every iteration -- see run_cold_route_class's docstring for why (hardware-forced,
    # not a design shortcut). ---
    for probe_level, selected_level in ROUTE_CLASSES:
        route_key = f"{probe_level}->{selected_level}"
        print(f"cold route {route_key}...")
        report["cold"][route_key] = run_cold_route_class(hef_paths, probe_level, selected_level, inputs)

    temp_end = read_temperature()
    report["temp_end_c"] = temp_end
    with hp.VDevice() as vdevice_temp_check:
        report["hailo_temp_end_c"] = read_hailo_temperature(vdevice_temp_check.get_physical_devices()[0])
    print(f"final temperature: {temp_end}")

    report["decision_microbenchmark"] = run_decision_microbenchmark()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
