# Hailo Dataflow Compiler: ONNX -> HAR -> HEF, all 4 levels — 2026-09-09

The last untested piece of the Week 1-2 compiler smoke test (`docs/RESEARCH_PLAN.md`
§5.1, `scripts/compiler_smoke_test.md`). Compiled on the researcher's own Windows
machine's WSL2 (Ubuntu 24.04, x86_64) using a Hailo Dataflow Compiler environment
already set up there from a prior unrelated project (`drone-rocket`) — not a new
install. No server or Hailo hardware needed for this step; only the target Hailo-8
runtime (device `E1`) needs the resulting `.hef`.

- git SHA: see `git log` at time of this report
- Model: `PaceSegSupernet` (random init, untrained), all four elasticity levels
- Toolchain: **Hailo Dataflow Compiler 3.34.0**, Hailo Model Zoo 2.19.0, Python 3.12.3,
  target `--hw-arch hailo8` (matches E1's confirmed real chip, see
  `SHARED_INFRASTRUCTURE.md` §3.2.1)
- Host: WSL2 on the researcher's Windows machine, AMD Ryzen 9 8945HX, no GPU used by
  DFC (WSL2 GPU passthrough not supported for DFC per Hailo's own docs — optimization
  ran at level 0, CPU only)

## Result: 4/4 parse -> optimize -> compile, all PASS

| Level | Parse (ONNX->HAR) | Optimize (HAR->HAR, random calib) | Compile (HAR->HEF) | HEF size |
|---|---|---|---|---|
| tiny | PASS | PASS | PASS | 743,414 B |
| small | PASS | PASS | PASS | 1,340,487 B |
| medium | PASS | PASS | PASS | 1,832,112 B |
| large | PASS | PASS | PASS | 3,144,773 B |

No unsupported-layer, parse-failure, or compile-failure messages anywhere across all
four levels' full logs (checked beyond exit codes this time). Every level produced a
real `.hef` file, ready to run on real Hailo-8 hardware via HailoRT.

Cluster utilization for `tiny` at compile time: total control 32.8%, compute 9.2%, memory
7.3% across 8 clusters -- plenty of headroom, consistent with this being a very small
(5,483-parameter) test model, not yet representative of a trained network's real size.

## What "optimize" here does NOT prove

`--use-random-calib-set` means the INT8 quantization ranges come from random/synthetic
data, not real images. This proves the **optimize and compile stages run mechanically**
for this architecture -- it says nothing about INT8 accuracy. The optimizer also printed
a real, useful hint (not an error): the calibration data was not normalized, and Hailo
recommends adding a normalization layer to the model so the neural core (not the host
CPU) performs normalization -- worth doing when real training/calibration data exists.

## Not yet done

- **Run the compiled HEF on real Hailo-8 hardware (E1).** E1 (`raspberrypi`,
  `100.93.114.43`) dropped off the visible Tailscale peer list during this session and
  could not be reached to copy the HEF over and test with `hailortcli run`. This is the
  next step once E1 is reachable again -- compiling is necessary but not sufficient;
  HailoRT actually loading and running the HEF is the real end-to-end confirmation.
- Calibrated (real-data) INT8 optimization
- `small`/`medium`/`large` were only parse+optimize+compile-tested, not yet run
- Any latency/energy number following the measurement protocol (§9)

## Reading

Combined with `E2_NX_compiler_smoke_test_20260908.md`, `E3_compiler_smoke_test_20260908.md`
and `E5_orin_nano_compiler_smoke_test_20260909.md`, this closes out the Week 1-2 gate from
`docs/SOURCE_RESEARCH_GAP_2026.md` §12 step 10 ("only start supernet training once at
least one graph has compiled through both TensorRT and Hailo end to end") for the
`PaceSegSupernet` architecture, at the compile level -- pending the E1 hardware run above
to fully close it.
