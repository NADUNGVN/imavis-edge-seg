# E3 (AGX Xavier) compiler smoke test — 2026-09-08

First real-hardware result for `docs/RESEARCH_PLAN.md` §5.1 / `scripts/compiler_smoke_test.md`.

- git SHA (source of the exported model): `8ba6f99a16d886427fc73691dcf94883a4e90e6e`
- Model: `PaceSegSupernet` (random init, untrained), `tiny` elasticity level, resolution 384x192, exported via `imavis_edge_seg.export.export_subnet_onnx` (ONNX opset 17, TorchScript exporter)
- Device: E3, Jetson AGX Xavier, `192.168.10.91`, L4T R35.6.4
- Toolchain: CUDA 11.4.19, TensorRT 8.5.2.2 (`trtexec`)

## TensorRT GPU (FP16)

```
trtexec --onnx=pace_seg_tiny.onnx --saveEngine=pace_seg_tiny_fp16.trt --fp16
```

Result: **PASSED**. No unsupported-operator or fallback warnings (only a benign
"subnormal FP16 values" weight-conversion warning, expected for untrained random
weights). GPU compute mean 1.25 ms, end-to-end mean latency 1.71 ms, p95 1.72 ms,
throughput ~797 qps (single stream, no other load on the device — not a representative
benchmark number, just confirms the engine runs).

## Xavier DLA (FP16, GPU fallback disabled)

```
trtexec --onnx=pace_seg_tiny.onnx --saveEngine=pace_seg_tiny_dla.trt --fp16 --useDLACore=0 --allowGPUFallback=false
```

Result: **PASSED** with `--allowGPUFallback=false`, meaning every layer in the graph
placed on DLA -- zero fallback to GPU. GPU-side wrapper compute mean 24.97 ms, end-to-end
mean latency 25.41 ms (DLA is expected to be slower than GPU for a model this small; the
number that matters here is that it compiled at all, not the latency).

## Reading

The compiler-safe operator set used in `src/imavis_edge_seg/models/blocks.py` (plain +
depthwise Conv2d, BatchNorm2d, ReLU, static-factor bilinear resize, elementwise add, a
1x1 classifier head with no final softmax) placed cleanly on **both** TensorRT GPU and
Xavier DLA with no unsupported layers, for this specific architecture at the `tiny`
elasticity level. This is a smoke test on random weights and one elasticity level only --
it does not yet confirm `small`/`medium`/`large`, does not confirm Hailo DFC compilation
(no Hailo Dataflow Compiler host set up yet -- HailoRT on E1 is the runtime, not the
compiler), and says nothing about accuracy or measured latency being representative
(single-stream, no thermal/power protocol followed, no baseline comparison).

## Not yet done

- `small`, `medium`, `large` elasticity levels on TensorRT/DLA
- Hailo DFC (ONNX -> HEF) -- needs the Dataflow Compiler installed on an x86 host, not
  yet set up anywhere
- INT8 (only FP16 tried here)
- Any latency/energy number that follows the measurement protocol in
  `docs/RESEARCH_PLAN.md` §9 (warm-up count, run count, power meter, thermal steady state)
