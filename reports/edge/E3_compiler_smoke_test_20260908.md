# E3 (AGX Xavier) compiler smoke test — 2026-09-08

First real-hardware result for `docs/RESEARCH_PLAN.md` §5.1 / `scripts/compiler_smoke_test.md`.

> **Correction (same day):** an earlier version of this report claimed "zero DLA
> fallback at every level" based on only reading the tail of one build log. A full
> re-check (verbose `trtexec` output, all four levels, both E3 and E2/NX) found real
> fallback at every level. The corrected finding below is more informative than the
> original wrong one — see "Root cause" section.

- git SHA (source of the exported models): `8ba6f99a16d886427fc73691dcf94883a4e90e6e`
- Model: `PaceSegSupernet` (random init, untrained), all four elasticity levels, each at
  its configured resolution, exported via `imavis_edge_seg.export.export_subnet_onnx`
  (ONNX opset 17, TorchScript exporter)
- Device: E3, Jetson AGX Xavier, `192.168.10.91`, L4T R35.6.4
- Toolchain: CUDA 11.4.19, TensorRT 8.5.2.2 (`trtexec`)

## GPU FP16 / GPU INT8 (uncalibrated): 8/8 PASS, no issues

| Level | Params | Resolution | GPU FP16 | GPU INT8* |
|---|---:|---|---|---|
| tiny | 5,483 | 384x192 | PASS, 780 qps | PASS, 755 qps |
| small | 20,611 | 512x256 | PASS, 406 qps | PASS, 285 qps |
| medium | 55,059 | 768x384 | PASS, 149 qps | PASS, 126 qps |
| large | 126,243 | 1024x512 | PASS, 68.3 qps | PASS, 64.5 qps |

\* INT8 here used `trtexec --int8` with **no calibration data** -- proves compilability
only, not accuracy.

## Xavier DLA (FP16): all 4 levels build, but with real fallback -- root cause identified

All four levels still produce a working engine (`trtexec` exits 0), but **not** with zero
fallback as first claimed. Layer-level fallback mentions in the log: tiny 41, small 52,
medium 64, large 88 (raw grep count, includes both real ops and auto-inserted
CAST/CONSTANT nodes).

**Root cause, confirmed directly from the TensorRT log** (not inferred):

```
[TRT] DLA supports only 16 subgraphs per DLA core. Switching to GPU for layer /fuse2/act_1/Relu
[TRT] DLA supports only 16 subgraphs per DLA core. Switching to GPU for layer /fuse1/act/Relu
...
```

Verbose output for the `tiny` level shows **16 `DlaLayer` fused subgraphs actually placed
on DLA** and 57 layers placed on GPU. The 16 DLA subgraphs cover the stem and all three
encoder stages (`stem`, `stage1.*`, `stage2.*`, `stage3.*` -- each fused
conv+BN+ReLU group becomes one DLA "ForeignNode" subgraph). The **decoder** (`reduce1-3`,
`fuse0-2`) never gets a DLA subgraph slot -- Xavier's DLA in this TensorRT version has a
**hard limit of 16 subgraphs per core**, and the decoder's resize-then-add-then-conv
pattern creates enough graph partition boundaries that the budget is exhausted before the
decoder is reached. This is not an "unsupported operator" problem (the same conv/BN/ReLU
ops run fine on DLA earlier in the same graph) -- it is a **subgraph-count budget**
problem caused by how the decoder is structured.

This is directly relevant to `docs/RESEARCH_PLAN.md`'s G2 compiler gap and RQ4
(compiler-constrained search space) -- it is exactly the kind of constraint the plan says
must be discovered by smoke-testing before finalizing the search space, not assumed.
**Actionable follow-up:** restructuring the decoder to fuse more of the
resize/add/conv sequence into fewer, larger blocks (e.g. avoid alternating small ops that
each become their own partition boundary) may recover DLA coverage for the decoder. Not
attempted yet.

Same finding reproduced on **E2 (Xavier NX)**, same TensorRT/DLA generation -- see
`E2_NX_compiler_smoke_test_20260908.md`.

## Reading

The op set in `src/imavis_edge_seg/models/blocks.py` compiles cleanly on TensorRT GPU
(FP16 and uncalibrated INT8) at every elasticity level on both E3 and E2. On DLA, the
**encoder** compiles and places on DLA cleanly; the **decoder currently does not** due to
a hardware subgraph-count limit, not an unsupported-operator issue. Do not claim "DLA
support" for the full model without either fixing the decoder's subgraph structure or
explicitly scoping DLA claims to the encoder.

## Not yet done

- Fixing the decoder's DLA subgraph count and re-testing
- Hailo DFC (ONNX -> HEF) -- needs the Dataflow Compiler installed on an x86 host, not
  yet set up anywhere; HailoRT on E1 is the runtime, not the compiler
- Jetson Nano -- not yet provided/reachable
- Real (calibrated) INT8 -- this run's INT8 had no calibration data
- Any latency/energy number that follows the measurement protocol in
  `docs/RESEARCH_PLAN.md` §9 (warm-up count, run count, power meter, thermal steady state)
- Training a real (non-random-weight) supernet, which is the actual prerequisite for any
  accuracy claim
