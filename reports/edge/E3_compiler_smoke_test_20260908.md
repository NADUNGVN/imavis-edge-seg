# E3 (AGX Xavier) compiler smoke test — 2026-09-08

First real-hardware result for `docs/RESEARCH_PLAN.md` §5.1 / `scripts/compiler_smoke_test.md`.

- git SHA (source of the exported models): `8ba6f99a16d886427fc73691dcf94883a4e90e6e`
- Model: `PaceSegSupernet` (random init, untrained), all four elasticity levels, each at
  its configured resolution, exported via `imavis_edge_seg.export.export_subnet_onnx`
  (ONNX opset 17, TorchScript exporter)
- Device: E3, Jetson AGX Xavier, `192.168.10.91`, L4T R35.6.4
- Toolchain: CUDA 11.4.19, TensorRT 8.5.2.2 (`trtexec`)

## Full matrix: 12/12 PASS

Ran `trtexec` for every (elasticity level) x (GPU-FP16, GPU-INT8, DLA-FP16 with GPU
fallback disabled) combination via `scripts/run_smoke_matrix.sh`, fed by ONNX exports
from `scripts/export_all_levels.py`.

| Level | Params | Resolution | GPU FP16 | GPU INT8* | DLA FP16 (0 fallback) |
|---|---:|---|---|---|---|
| tiny | 5,483 | 384x192 | PASS, 780 qps | PASS, 755 qps | PASS, 39.8 qps |
| small | 20,611 | 512x256 | PASS, 406 qps | PASS, 285 qps | PASS, 24.2 qps |
| medium | 55,059 | 768x384 | PASS, 149 qps | PASS, 126 qps | PASS, 6.4 qps |
| large | 126,243 | 1024x512 | PASS, 68.3 qps | PASS, 64.5 qps | PASS, 3.5 qps |

\* INT8 here used `trtexec --int8` with **no calibration data** (`Calibrator is not being
used. Users must provide dynamic range for all tensors...` warning present in every INT8
run) -- this only proves the graph *compiles* in INT8 mode, it says nothing about INT8
*accuracy*. Real INT8 numbers require the calibration set described in
`docs/RESEARCH_PLAN.md` §5.2 (day/night/rain/fog/snow) and QAT/PTQ per §6.2, not this
smoke test.

Every DLA build used `--allowGPUFallback=false` and still passed at every level --
zero layers fell back to GPU across the whole matrix. DLA throughput is far below GPU at
every size (e.g. tiny: 39.8 vs 780 qps) -- expected for models this small (5K-126K
params; DLA has fixed per-invocation overhead that dominates until the model is much
larger), not a claim that DLA is slower in general. Single-stream, no thermal/power
protocol, one run per config -- not a benchmark number, only a compile+run confirmation.

No unsupported-operator or fallback errors anywhere in the matrix; the only recurring
warning was the benign FP16 "subnormal weight" notice (expected for untrained random
weights) and the expected INT8 no-calibrator notice above.

## Reading

The compiler-safe operator set used in `src/imavis_edge_seg/models/blocks.py` (plain +
depthwise Conv2d, BatchNorm2d, ReLU, static-factor bilinear resize, elementwise add, a
1x1 classifier head with no final softmax) placed cleanly on **both** TensorRT GPU and
Xavier DLA with no unsupported layers, for this specific architecture, at **all four**
elasticity levels, in both FP16 and (uncalibrated) INT8 build modes. This is a strong
signal that the chosen op set is a valid intersection for TensorRT-family backends across
the whole elastic search space (RESEARCH_PLAN.md contribution 1's "candidate
intersection" claim), on E3 specifically. It is still a smoke test on random weights,
still single-device (E3 only, not Xavier NX or Nano), and says nothing about accuracy or
measured latency/energy being representative (single-stream, no thermal/power protocol,
one run per config, no baseline comparison).

## Not yet done

- Hailo DFC (ONNX -> HEF) -- needs the Dataflow Compiler installed on an x86 host, not
  yet set up anywhere; HailoRT on E1 is the runtime, not the compiler
- Xavier NX (E2) and Jetson Nano -- not yet provided/reachable
- Real (calibrated) INT8 -- this run's INT8 had no calibration data, so it only proves
  compilability, not accuracy
- Any latency/energy number that follows the measurement protocol in
  `docs/RESEARCH_PLAN.md` §9 (warm-up count, run count, power meter, thermal steady state)
- Training a real (non-random-weight) supernet, which is the actual prerequisite for any
  accuracy claim
