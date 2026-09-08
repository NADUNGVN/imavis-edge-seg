# E2 (Jetson Xavier NX) compiler smoke test — 2026-09-08

Same method as `E3_compiler_smoke_test_20260908.md` (read that file first for the DLA
16-subgraph-per-core root cause, confirmed on both devices). This file records E2's own
numbers only.

- git SHA: `8ba6f99a16d886427fc73691dcf94883a4e90e6e`
- Device: E2, **NVIDIA Jetson Xavier NX Developer Kit**, `192.168.10.93`, L4T R35.4.1
- Toolchain: CUDA 11.4.19, cuDNN 8.6.0.166, TensorRT 8.5.2.2 (`trtexec`) -- **already
  installed** on this device before this session touched it (unlike E3, which needed
  installing)
- Real hostname: `arar-desktop`; CPU shows 6 cores total, 2 offline (`Carmel`, matches NX
  spec of 6 cores vs AGX Xavier's 8)

## GPU FP16 / GPU INT8 (uncalibrated) / DLA FP16: 12/12 build (trtexec exit 0)

| Level | Params | Resolution | GPU FP16 | GPU INT8* | DLA FP16** |
|---|---:|---|---|---|---|
| tiny | 5,483 | 384x192 | PASS, 581 qps | PASS, 523 qps | PASS, 118 qps |
| small | 20,611 | 512x256 | PASS, 267 qps | PASS, 231 qps | PASS, 67.9 qps |
| medium | 55,059 | 768x384 | PASS, 98.5 qps | PASS, 82.4 qps | PASS, 19.1 qps |
| large | 126,243 | 1024x512 | PASS, 43.5 qps | PASS, 41.5 qps | PASS, 9.9 qps |

\* No calibration data -- compilability only. \*\* Same 16-subgraph-per-DLA-core fallback
as E3: encoder runs on DLA, decoder falls back to GPU. Fallback layer-mention counts
(tiny/small/medium/large): 41/52/64/88 -- identical to E3, as expected since it's the same
graph structure and the same TensorRT/DLA generation.

GPU throughput on NX is lower than E3 at every level (expected -- NX is the smaller/lower
-power Xavier SKU), but DLA throughput is notably *higher* than E3 at `tiny`
(118 vs 39.8 qps) despite NX being the weaker device overall -- not yet explained; could
be measurement noise (single run, no thermal/power protocol) or a genuine NX-vs-AGX DLA
clocking difference. Not investigated further; flagging so it isn't silently treated as
consistent behavior across devices.

## Not yet done

Same list as `E3_compiler_smoke_test_20260908.md`, plus: NX's `nvpmodel` power mode was
not queried for either device during this run.
