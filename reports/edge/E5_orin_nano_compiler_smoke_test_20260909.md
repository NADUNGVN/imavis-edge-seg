# E5 (Jetson Orin Nano Super) compiler smoke test — 2026-09-09

Same method as the E2/E3 reports (see those for the DLA 16-subgraph-per-core finding on
Xavier-generation devices — does not apply here, this device has no DLA at all).

- git SHA: `55c1f89` (see `git log` for the exact commit this was run against)
- Device: **E5**, NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super,
  `100.101.232.97` (Tailscale, relayed — ~230ms ping, not LAN), hostname `ubuntu`, user `huy`
- **This is not the legacy "Jetson Nano" (Maxwell GPU, JetPack 4, EOL) the original
  research plan assumed.** It is a current-generation **Orin Nano Super**: L4T R36.5.0
  (JetPack 6.x generation), Ubuntu 22.04.5, Cortex-A78AE CPU, NVMe SSD (915G, not
  eMMC/microSD), power modes 15W/25W/MAXN_SUPER (the "Super" firmware unlock).
- Toolchain: CUDA 12.6.68, cuDNN 9.3.0 (cuda12 build), TensorRT **10.3.0.30** — all
  already installed, none installed by this session. Notably newer major versions than
  E2/E3's CUDA 11.4/TensorRT 8.5.2.
- **No DLA on this device** — confirmed empirically: `trtexec --useDLACore=0` fails with
  `Cannot create DLA engine, 0 not available`. This matches the general Orin Nano vs Orin
  NX/AGX Orin distinction (Nano SKU has DLA disabled/absent); not re-verified against
  NVIDIA's official spec sheet, just observed directly on this unit.

## GPU FP16 / GPU INT8 (uncalibrated): 8/8 PASS, no unsupported ops

| Level | Params | Resolution | GPU FP16 | GPU INT8* |
|---|---:|---|---|---|
| tiny | 5,483 | 384x192 | PASS, 946 qps | PASS, 799 qps |
| small | 20,611 | 512x256 | PASS, 678 qps | PASS, 468 qps |
| medium | 55,059 | 768x384 | PASS, 189 qps | PASS, 199 qps |
| large | 126,243 | 1024x512 | PASS, 83.4 qps | PASS, 125 qps |

\* No calibration data — proves compilability only, not accuracy. Note INT8 throughput
exceeds FP16 at medium/large (unlike E2/E3) -- plausible given Orin's newer/faster native
INT8 path, not investigated further; single run, not a benchmark number.

No unsupported-operator, fallback, or error messages in the full log (checked beyond just
the tail this time, learning from the E2/E3 mistake).

## Reading

Same op set (`src/imavis_edge_seg/models/blocks.py`) compiles cleanly on TensorRT GPU on
a third, newer Jetson generation (Orin) in addition to the two Xavier-generation devices.
No DLA path exists to test here. This device's CUDA/TensorRT major-version gap from
E2/E3 (12.6/TRT10 vs 11.4/TRT8.5) is itself a relevant data point for
`docs/RESEARCH_PLAN.md`'s "compiler-version-dependent results" limitation -- engines
built for E2/E3 will very likely not load on E5 and vice versa; each device needs its own
compiled engine per subnet, which the project's static-engine-per-backend design already
assumes.

## Not yet done

- Hailo DFC, Jetson Nano (the actual legacy board, if it still needs separate coverage)
- Calibrated INT8
- Latency/energy numbers following the measurement protocol (§9)
- Confirming whether the "Jetson Nano" row in the original hardware plan should be
  considered fulfilled by this device, or whether the legacy Nano is still wanted
  separately -- see `docs/INFRA_OVERRIDE.md`
