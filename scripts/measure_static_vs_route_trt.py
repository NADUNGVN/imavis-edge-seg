"""Static vs adaptive-route latency on TensorRT with ONE harness and ONE timer boundary
(review round 2, 2026-10-03; run on E3, optionally E2/E5).

Why: the manuscript compared routes timed by `measure_router_overhead.py` (Python
harness, full-logit D2H copy) with static latencies from `trtexec`. Those are two
different harnesses, so neither the static reference nor the route cost is directly
comparable. This script times, in the same process, with the same engines, inputs,
copies and synchronization:

  static_<level>      : one engine, H2D input -> execute -> output handling
  route_tiny-><level> : tiny probe (H2D + execute) -> GPU entropy -> policy-D decision
                        -> (if level != tiny) selected engine H2D -> execute
                        -> output handling of the returned prediction

Output handling (`--output-mode`, both by default):
  logits : copy the full float32 logits to host (what the old harness did)
  none   : leave the prediction on the device (lower bound, e.g. when the next stage
           runs on the GPU)
The tiny route returns the probe's own prediction and applies the same output mode.

Every class is measured warm (all engines resident), interleaved in a randomized
order, 50 warm-up + >=500 timed iterations, with temperature, nvpmodel and clocks
recorded. Inputs are pre-generated host tensors at each level's resolution (resize
and normalization are outside the timer, as in the paper).

Usage (on the Jetson, in the project env):
  sudo nvpmodel -q > reports/edge/E3_nvpmodel_$(date +%Y%m%d).txt
  python scripts/measure_static_vs_route_trt.py --engine-dir <dir with tiny/small/medium/large .engine> \
      --device-label E3 --output-json reports/static_vs_route_E3_<date>.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt
from measure_router_overhead import (
    LEVELS,
    TrtEngine,
    decision_d,
    fit_fake_calibrator,
    read_temperature,
    summarize,
)

WARMUP = 50
MEASURED = 500


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception as exc:  # recorded, never fabricated
        return f"unavailable: {exc}"


def system_state() -> dict[str, Any]:
    gov = None
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if p.exists():
        gov = p.read_text().strip()
    return {
        "platform": platform.platform(),
        "nvpmodel": _cmd(["nvpmodel", "-q"]),
        "jetson_clocks_show": _cmd(["jetson_clocks", "--show"]),
        "cpu0_governor": gov,
        "tensorrt_version": trt.__version__,
    }


def output_step(engine: TrtEngine, mode: str) -> None:
    if mode == "logits":
        import pycuda.driver as cuda

        cuda.memcpy_dtoh_async(engine.host_out, engine.device_out, engine.stream)
    engine.stream.synchronize()


def run_static(engine: TrtEngine, x: np.ndarray, mode: str) -> None:
    engine.infer_no_copy(x)  # H2D + execute + sync
    output_step(engine, mode)


def run_route(engines: dict[str, TrtEngine], inputs: dict[str, np.ndarray], target: str, mode: str,
              cals: dict, costs: dict[str, float]) -> None:
    probe = engines["tiny"]
    probe.infer_no_copy(inputs["tiny"])
    score = probe.risk_score_gpu()
    _ = decision_d(score, cals, 0.1, max(costs.values()), costs)  # decision cost is paid every frame
    if target == "tiny":
        output_step(probe, mode)
    else:
        engines[target].infer_no_copy(inputs[target])
        output_step(engines[target], mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-dir", type=Path, required=True)
    ap.add_argument("--engine-pattern", default="pace_seg_{level}.engine", help="file name pattern inside --engine-dir")
    ap.add_argument("--device-label", required=True)
    ap.add_argument("--output-mode", choices=["logits", "none", "both"], default="both")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--iters", type=int, default=MEASURED)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    engines = {lv: TrtEngine(args.engine_dir / args.engine_pattern.format(level=lv)) for lv in LEVELS}
    rng = np.random.default_rng(42)
    inputs = {}
    for lv, eng in engines.items():
        shape = next(tuple(eng.engine.get_binding_shape(i)) for i in range(eng.engine.num_bindings)
                     if eng.engine.binding_is_input(i))
        inputs[lv] = rng.standard_normal(shape, dtype=np.float32)
    cal_rng = np.random.default_rng(0)
    cals = {lv: fit_fake_calibrator(cal_rng) for lv in LEVELS}
    placeholder_costs = {"tiny": 1.0, "small": 2.0, "medium": 4.0, "large": 8.0}

    modes = ["logits", "none"] if args.output_mode == "both" else [args.output_mode]
    classes = [(kind, lv, m) for m in modes for kind in ("static", "route") for lv in LEVELS]
    random.Random(0).shuffle(classes)

    report: dict[str, Any] = {"device_label": args.device_label, "system": system_state(),
                              "temp_start_c": read_temperature(args.device_label),
                              "protocol": {"warmup": args.warmup, "iters": args.iters,
                                           "timer": "host tensor H2D start -> synchronized output (mode-dependent)"},
                              "results": {}}
    for kind, lv, mode in classes:
        key = f"{kind}_{'tiny->' if kind == 'route' else ''}{lv}|{mode}"
        samples = []
        for i in range(args.warmup + args.iters):
            t0 = time.perf_counter()
            if kind == "static":
                run_static(engines[lv], inputs[lv], mode)
            else:
                run_route(engines, inputs, lv, mode, cals, placeholder_costs)
            t1 = time.perf_counter()
            if i >= args.warmup:
                samples.append((t1 - t0) * 1000.0)
        report["results"][key] = summarize(samples)
        print(f"{key}: median {report['results'][key]['median_ms']:.3f} ms  p99 {report['results'][key]['p99_ms']:.3f}")
    report["temp_end_c"] = read_temperature(args.device_label)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
