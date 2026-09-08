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

2026-09-08 result on **E3 (AGX Xavier)**, `PaceSegSupernet` `tiny` subnet (own
architecture, not Fast-SCNN — see `../reports/edge/E3_compiler_smoke_test_20260908.md`
for full detail and caveats):

| Operator | TensorRT GPU | Xavier DLA | Hailo HEF | Notes |
|---|---|---|---|---|
| Conv (depthwise) | **PASS** (FP16, 2026-09-08) | **PASS** (FP16, 0 fallback, 2026-09-08) | not tested | Hailo DFC not installed anywhere yet |
| Conv (pointwise) | **PASS** | **PASS** | not tested | |
| BatchNorm (fused) | **PASS** | **PASS** | not tested | |
| ReLU / ReLU6 | **PASS** (ReLU only, not ReLU6) | **PASS** | not tested | |
| Static resize (bilinear) | **PASS** | **PASS** | not tested | |
| Add / Concat | **PASS** (Add only, not Concat) | **PASS** | not tested | |
| Softmax (final) | N/A — not used | N/A — not used | not tested | Classifier head outputs raw logits, no final softmax, by design; avoids the known DLA softmax restriction entirely |

Still not tested anywhere: `small`/`medium`/`large` elasticity levels, INT8, Hailo DFC
(needs an x86 host with the Dataflow Compiler installed — not set up yet), Xavier NX,
Jetson Nano.

## Gate

Only start supernet training (Phase 4 in `README.md`) once at least one graph has
compiled through **both** TensorRT and Hailo end to end (`RESEARCH_PLAN.md` source
doc §12, step 10).
