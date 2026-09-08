# Compiler smoke test (Week 1-2 gate)

Purpose: find the real compiler-safe operator intersection across TensorRT GPU, Xavier
DLA and Hailo DFC *before* the supernet search space is finalized
(`docs/RESEARCH_PLAN.md` §5.1, §13). Do not assume any operator behaves identically
across backends without a result in the table below.

## Steps

1. Export one small reference model (Fast-SCNN or BiSeNetV2) to ONNX, static shape,
   batch 1.
2. **TensorRT (Nano/NX/AGX):** build FP16 and INT8 engines. Record success/failure and
   any operator that falls back to plugin/unsupported.
3. **Xavier DLA (NX/AGX):** build a DLA-targeted engine with GPU fallback **disabled
   first** to surface unsupported layers, then re-enable fallback and record which
   layers actually ran off-DLA.
4. **Hailo (E1):** run ONNX → Hailo Dataflow Compiler → HEF. Record parse/optimize/
   compile failures per layer.
5. Fill in the table below and paste it into `docs/INFRA_OVERRIDE.md` or a dated report
   under `../reports/` (create that folder when the first real result exists).

## Result table (fill in per run)

| Operator | TensorRT GPU | Xavier DLA | Hailo HEF | Notes |
|---|---|---|---|---|
| Conv (depthwise) | | | | |
| Conv (pointwise) | | | | |
| BatchNorm (fused) | | | | |
| ReLU / ReLU6 | | | | |
| Static resize (bilinear) | | | | |
| Add / Concat | | | | |
| Softmax (final) | | | | DLA does not support softmax — confirm plugin/host fallback |

## Gate

Only start supernet training (Phase 4 in `README.md`) once at least one graph has
compiled through **both** TensorRT and Hailo end to end (`RESEARCH_PLAN.md` source
doc §12, step 10).
