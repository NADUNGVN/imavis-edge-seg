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

2026-09-08 result on **E3 (AGX Xavier) and E2 (Xavier NX)**, `PaceSegSupernet` at **all
four elasticity levels** (own architecture, not Fast-SCNN — see
`../reports/edge/E3_compiler_smoke_test_20260908.md` and
`../reports/edge/E2_NX_compiler_smoke_test_20260908.md` for full per-level latency and
an important correction to an earlier wrong "0 fallback" claim). Reproduce with
`export_all_levels.py` + `run_smoke_matrix.sh`:

| Operator | TensorRT GPU (FP16 + INT8*) | Xavier DLA (FP16) | Hailo HEF | Notes |
|---|---|---|---|---|
| Conv (depthwise) | **PASS**, all 4 levels, E2+E3 | **encoder: DLA. decoder: GPU fallback** (both devices, all levels) — see root cause below | not tested | Hailo DFC not installed anywhere yet |
| Conv (pointwise) | **PASS** | same split | not tested | |
| BatchNorm (fused) | **PASS** | same split | not tested | |
| ReLU / ReLU6 | **PASS** (ReLU only, not ReLU6) | same split | not tested | |
| Static resize (bilinear) | **PASS** | in the fallback region (decoder) | not tested | |
| Add / Concat | **PASS** (Add only, not Concat) | in the fallback region (decoder) | not tested | |
| Softmax (final) | N/A — not used | N/A — not used | not tested | Classifier head outputs raw logits, no final softmax, by design; avoids the known DLA softmax restriction entirely |

\* INT8 here had no calibration data — proves compilability only, not accuracy.

**DLA root cause (confirmed from TensorRT's own log, both E2 and E3):** *"DLA supports
only 16 subgraphs per DLA core"*. The encoder (stem + 3 stages) fits within that budget
and runs on DLA; the decoder's resize→add→conv pattern creates enough extra subgraph
partitions that it exhausts the budget and falls back to GPU wholesale. This is a
subgraph-count limit, not an unsupported-operator problem — restructuring the decoder to
reduce partition count is a plausible fix, not yet attempted.

Still not tested anywhere: Hailo DFC (needs an x86 host with the Dataflow Compiler
installed — not set up yet), Jetson Nano, calibrated INT8, a decoder redesign that fits
the 16-subgraph DLA budget.

## Gate

Only start supernet training (Phase 4 in `README.md`) once at least one graph has
compiled through **both** TensorRT and Hailo end to end (`RESEARCH_PLAN.md` source
doc §12, step 10).
