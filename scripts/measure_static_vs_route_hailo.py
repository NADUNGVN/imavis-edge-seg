"""Static vs adaptive-route latency on Hailo-8 (E1) with ONE harness and ONE timer
boundary, for two runtime paths (review round 2, 2026-10-03).

Why: the manuscript's E1 route costs come from `measure_router_overhead_hailo.py`,
which activates/deactivates each network group per call and uses FLOAT32 vstreams
(host-side quantize/dequantize), while the static reference came from `hailortcli`
with a continuously activated group. This script measures both static and route
classes through the same InferVStreams pipelines:

  --path explicit  : single-context VDevice, no scheduler. Static classes activate the
                     candidate's network group ONCE and run it repeatedly (what a static
                     deployment does); route classes activate/deactivate per call (the
                     paper's current route harness).
  --path scheduler : VDevice created with the HailoRT model scheduler (round robin);
                     no explicit activation -- the scheduler switches network groups.
                     NOTE: verify on HailoRT 4.23 that InferVStreams without activate()
                     is supported with the scheduler; if this path raises, record the
                     error text in the output JSON and do not substitute numbers.

  --vstream-format float32|uint8 : FLOAT32 vstreams include host-side
                     (de)quantization; UINT8 isolates device + transfer time. The
                     entropy probe needs float logits, so for uint8 the tiny output is
                     dequantized on the host inside the timer for route classes.

Timer: from handing the pre-generated host input to the pipeline to receiving the
selected output on the host. Run each path separately (one VDevice at a time).

Usage (on the Raspberry Pi 5 + Hailo-8):
  python scripts/measure_static_vs_route_hailo.py --hef-dir <dir> --path explicit \
      --vstream-format float32 --output-json reports/static_vs_route_E1_explicit_f32_<date>.json
  ... repeat for --path scheduler and --vstream-format uint8
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import hailo_platform as hp
import numpy as np
from measure_router_overhead_hailo import (
    LEVELS,
    decision_d,
    fit_fake_calibrator,
    read_hailo_temperature,
    read_temperature,
    summarize,
)

WARMUP = 50
MEASURED = 500


def entropy_from_output(out: np.ndarray, quant: tuple[float, float] | None) -> float:
    x = out.astype(np.float32)
    if quant is not None:
        scale, zp = quant
        x = (x - zp) * scale
    x = x.reshape(-1, x.shape[-1])  # (H*W, C), NHWC output
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum(axis=1).mean())


class Pipe:
    def __init__(self, vdevice: Any, hef_path: Path, fmt: Any, stack: ExitStack) -> None:
        self.hef = hp.HEF(str(hef_path))
        params = hp.ConfigureParams.create_from_hef(self.hef, interface=hp.HailoStreamInterface.PCIe)
        self.ng = vdevice.configure(self.hef, params)[0]
        self.ng_params = self.ng.create_params()
        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_info = self.hef.get_output_vstream_infos()[0]
        ip = hp.InputVStreamParams.make(self.ng, format_type=fmt)
        op = hp.OutputVStreamParams.make(self.ng, format_type=fmt)
        self.pipe = stack.enter_context(hp.InferVStreams(self.ng, ip, op))
        qi = self.out_info.quant_info
        self.quant = (float(qi.qp_scale), float(qi.qp_zp)) if fmt == hp.FormatType.UINT8 else None

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.pipe.infer({self.in_info.name: x})[self.out_info.name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hef-dir", type=Path, required=True)
    ap.add_argument("--path", choices=["explicit", "scheduler"], required=True)
    ap.add_argument("--vstream-format", choices=["float32", "uint8"], default="float32")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--iters", type=int, default=MEASURED)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    fmt = hp.FormatType.FLOAT32 if args.vstream_format == "float32" else hp.FormatType.UINT8
    report: dict[str, Any] = {"device_label": "E1", "path": args.path, "vstream_format": args.vstream_format,
                              "hailort_version": getattr(hp, "__version__", "unknown"),
                              "temp_start_c": read_temperature(), "results": {}, "errors": {}}
    rng = np.random.default_rng(42)
    cal_rng = np.random.default_rng(0)
    cals = {lv: fit_fake_calibrator(cal_rng) for lv in LEVELS}
    costs = {"tiny": 1.0, "small": 2.0, "medium": 4.0, "large": 8.0}

    vparams = hp.VDevice.create_params()
    if args.path == "scheduler":
        vparams.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN
    try:
        with hp.VDevice(vparams) as vdevice, ExitStack() as stack:
            report["hailo_temp_start_c"] = read_hailo_temperature(vdevice.get_physical_devices()[0])
            pipes = {lv: Pipe(vdevice, args.hef_dir / f"pace_seg_{lv}.hef", fmt, stack) for lv in LEVELS}
            dtype = np.float32 if fmt == hp.FormatType.FLOAT32 else np.uint8
            inputs = {}
            for lv, p in pipes.items():
                shape = (1, *p.in_info.shape)
                inputs[lv] = (rng.standard_normal(shape).astype(np.float32) if dtype == np.float32
                              else rng.integers(0, 255, shape, dtype=np.uint8))

            def run(pipe: Pipe, x: np.ndarray, activate: bool) -> np.ndarray:
                if activate:
                    with pipe.ng.activate(pipe.ng_params):
                        return pipe.infer(x)
                return pipe.infer(x)

            classes = [(k, lv) for k in ("static", "route") for lv in LEVELS]
            random.Random(0).shuffle(classes)
            for kind, lv in classes:
                key = f"{kind}_{'tiny->' if kind == 'route' else ''}{lv}"
                samples: list[float] = []
                try:
                    if kind == "static" and args.path == "explicit":
                        # static deployment: activate once, then run continuously
                        with pipes[lv].ng.activate(pipes[lv].ng_params):
                            for i in range(args.warmup + args.iters):
                                t0 = time.perf_counter()
                                pipes[lv].infer(inputs[lv])
                                t1 = time.perf_counter()
                                if i >= args.warmup:
                                    samples.append((t1 - t0) * 1000.0)
                    else:
                        act = args.path == "explicit"
                        for i in range(args.warmup + args.iters):
                            t0 = time.perf_counter()
                            if kind == "static":
                                run(pipes[lv], inputs[lv], act)
                            else:
                                out = run(pipes["tiny"], inputs["tiny"], act)
                                score = entropy_from_output(out, pipes["tiny"].quant)
                                _ = decision_d(score, cals, 0.1, 8.0, costs)
                                if lv != "tiny":
                                    run(pipes[lv], inputs[lv], act)
                            t1 = time.perf_counter()
                            if i >= args.warmup:
                                samples.append((t1 - t0) * 1000.0)
                    report["results"][key] = summarize(samples)
                    print(f"{key}: median {report['results'][key]['median_ms']:.2f} ms  p99 {report['results'][key]['p99_ms']:.2f}")
                except Exception:
                    report["errors"][key] = traceback.format_exc()
                    print(f"{key}: ERROR (recorded)")
            report["hailo_temp_end_c"] = read_hailo_temperature(vdevice.get_physical_devices()[0])
    except Exception:
        report["errors"]["setup"] = traceback.format_exc()
        print("setup ERROR (recorded)")
    report["temp_end_c"] = read_temperature()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
