# PACE-Seg — research plan

> Source: adapted from `docs/SOURCE_RESEARCH_GAP_2026.md` (deep-research report,
> sources locked 2026-08-24). This file is the living plan; the source file is kept as
> an immutable historical reference and should not be edited.

## 1. One-sentence pitch

Train one elastic segmentation supernet; extract 3-4 compiler-compatible static INT8
subnets; route between them per image/time-window using a calibrated difficulty/risk
probe; optimize on **measured p95 latency and J/frame**, not FLOPs; verify on Jetson
Nano, Xavier NX, AGX Xavier and Pi 5 + Hailo-8 under day, night, rain, fog and snow.

## 2. Why this direction (not a FLOPs-only lightweight model, not a YOLO+attention
variant, not a pure FPS benchmark table)

IMAVIS wants a CV method contribution; edge deployment is evidence for it, not the whole
contribution. The defensible gap sits at the intersection of four axes, each individually
already covered in recent literature (lightweight CNN reviews, YOLIC, Light-SEF, UCPNet,
HARD, dynamic-network surveys — see source doc §2 for the literature table):

- **G1 — Proxy gap**: params/FLOPs do not reliably predict latency/energy across
  different accelerator architectures.
- **G2 — Compiler gap**: a graph that runs on TensorRT GPU is not guaranteed to run
  (unmodified) on Xavier DLA or compile to a Hailo HEF (DLA has no dynamic shapes, no
  softmax; GPU fallback can silently hide non-DLA layers).
- **G3 — Reliability gap**: clean-image mIoU does not reflect risk under night/rain/
  fog/snow; ACDC provides uncertain-region annotations that enable risk-aware evaluation
  instead of average mIoU alone.
- **G4 — Static-budget gap**: one independently-trained model per device wastes
  training/maintenance cost and cannot exploit budget that varies with device, power
  mode, thermal state and per-image difficulty.

**Claim we are testing**: a single segmentation method that (a) generates static engines
compatible with multiple accelerators, (b) allocates compute by calibrated visual risk,
and (c) is optimized on measured latency + energy under adverse domain shift is not
convincingly solved by the literature surveyed so far. Do not write "the first" in the
manuscript before a full Scopus/Web of Science systematic search close to submission.

## 3. Research questions and testable hypotheses

| RQ | Question | Hypothesis |
|---|---|---|
| RQ1 | Does measured-hardware-aware search beat FLOPs-aware search? | At equal mIoU, a measured p95-latency/J-frame objective cuts cost by ≥15-20% vs a FLOPs-constrained search, on most devices. |
| RQ2 | Can one supernet replace several independently-trained models? | Subnets extracted from the supernet land within 1.0-1.5 mIoU of independently-trained models at the same budget, at much lower training/maintenance cost. |
| RQ3 | Does calibrated difficulty routing help under adverse shift? | Routing hits a risk/accuracy target at lower energy than static-large, and beats entropy-only routing on AURC/UIoU. |
| RQ4 | Does a compiler-constrained search space actually improve portability? | Higher compile-success rate / accelerator coverage, fewer fallbacks, lower latency variance than an unconstrained operator space. |

## 4. Four intended contributions

1. **Compiler-safe elastic segmentation supernet** — elastic in width, depth, input
   resolution; search space restricted to the operator intersection across target
   compilers. This intersection is a *candidate* set until smoke-tested per compiler in
   weeks 1-2 — never assume Conv/Resize behave identically across TensorRT/DLA/Hailo DFC.
2. **Hardware-in-the-loop Pareto search** — a latency/energy lookup table or surrogate
   per backend built from real measurements; subnet selection uses multi-objective
   measured cost, not FLOPs.
3. **Calibrated visual-risk router** — a small quality/risk probe predicts expected
   segmentation error from a downsampled image; picks one of the compiled static engines
   under a risk target and energy budget. Calibration is fit on validation data only.
4. **Cross-platform reliability benchmark protocol** — clean/adverse accuracy,
   uncertainty, p50/p95/p99 latency, J/frame, thermal stability, compiler coverage across
   four platforms with full artifact/version disclosure.

## 5. Method sketch

### 5.1 Supernet

- Backbone: simple CNN/hybrid blocks — Conv-BN-act, depthwise/pointwise conv, pooling,
  add/concat, static resize. Avoid dynamic runtime tensor shapes.
- Four elasticity levels: `tiny`, `small`, `medium`, `large`.
- Elastic axes: channel multiplier, block count, input resolution.
- Sandwich rule + in-place distillation for shared-weight training.
- Boundary-aware auxiliary loss to protect pedestrian/pole/sign/road-boundary classes at
  small subnet sizes.

### 5.2 Quantization

- FP32 teacher → shared supernet → QAT INT8.
- Calibration set must represent day/night/rain/fog/snow.
- Report FP32, FP16, INT8 separately — never merge pre/post-quantization accuracy.
- Hailo path: ONNX/TF → Dataflow Compiler → HEF; DFC's own profiler/emulator numbers are
  never a substitute for on-chip measurement.

### 5.3 Static engines, not a dynamic graph

- Compile each subnet to: a TensorRT engine (GPU), a DLA-compatible engine (Xavier), and
  a Hailo HEF.
- Router runs on host (or a small accelerator), switches engine every 8-32 frame window
  to amortize switching overhead — no dynamic control flow inside any single graph.

### 5.4 Objective

```
min_{a,theta}  L_seg + alpha * L_distill + beta * L_cal
               + lambda * L_hat_p95(a, device, precision)
               + mu     * E_hat(a, device, precision)
```
subject to: compile success, memory budget, accelerator coverage, risk target.
`L_hat` / `E_hat` are surrogates fit from measured p95 latency and energy/frame, not FLOPs.

## 6. Datasets

| Role | Dataset | Use |
|---|---|---|
| Clean / source | Cityscapes (5k fine, 20k coarse, 50 cities) | Train backbone; clean mIoU + real-time benchmark |
| Adverse (primary) | ACDC (4,006 images, fog/night/rain/snow, uncertainty masks) | Official train/val/test split; per-condition + uncertainty-aware eval |
| External shift | Dark Zurich | Zero-shot / adaptation-free external test; never tune thresholds on it |
| Optional | BDD100K | Only if compute allows; lower priority than depth of ablation |

**Minimum scope: Cityscapes + ACDC.** Do not add datasets at the cost of ablation depth.

## 7. Required baselines

Fast-SCNN, BiSeNetV2, PIDNet-S or DDRNet-23-slim, SegFormer-B0,
MobileNetV3-Large+DeepLabV3, HARD (if code/checkpoint reproducible), UCPNet (if released
in time).

Baseline axes: independent-per-budget training vs shared supernet; FLOPs-aware vs
latency-aware vs latency+energy-aware selection; static-small/static-large/oracle
router/entropy router/calibrated risk router; FP32/FP16/INT8-PTQ/INT8-QAT;
unconstrained vs compiler-constrained operator space.

## 8. Metrics

**Vision quality**: overall + per-condition mIoU; per-class IoU (person/rider,
traffic sign/light, pole, road boundary); boundary IoU/F-score; FP32→FP16/INT8 drop.

**Reliability**: ECE, NLL/Brier, risk-coverage curve + AURC, UIoU on ACDC/Dark Zurich,
selective mIoU at fixed coverage/risk targets.

**Deployment**: p50/p95/p99 latency (batch 1), single-stream throughput, J/frame and
images/J, idle/active system power, peak RAM/device memory, thermal/clock/throttle over
sustained runs, compile success + operator fallback rate, preprocess/transfer/
inference/postprocess breakdown.

## 9. Power/latency protocol (borrow MLPerf Power principles, no compliance claim)

1. Batch 1; identical resolution/pre/postprocessing across configs.
2. Warm-up ≥200 inferences; measure ≥5,000 frames or until CI is stable.
3. Three independent runs per config; randomize model order to reduce thermal bias.
4. One sustained 30-60 min run per representative model.
5. Lock and disclose `nvpmodel`, clocks, fan mode, ambient temp, OS, JetPack, CUDA,
   TensorRT, HailoRT, DFC, firmware.
6. End-to-end power via external calibrated meter; `tegrastats`/telemetry explains,
   never replaces, the system-level number.
7. Report kernel-only and end-to-end separately; never compare Jetson kernel-only to
   Pi end-to-end.
8. Bootstrap 95% CI for latency/energy; 3 training seeds for headline accuracy numbers.

## 10. Ablations

Shared supernet vs independent training; FLOPs vs measured-latency objective;
latency-only vs latency+energy; unconstrained vs compiler-safe operator space; PTQ vs
QAT; with/without distillation; static-large/small vs router; entropy vs calibrated
risk router; per-frame vs temporal-window routing; behavior across 5/10/15/30 W power
modes.

## 11. Go/no-go criteria (fixed in advance, no post-hoc cherry-picking)

**Go**
- All four devices run ≥2 static INT8 subnets; NX/AGX have GPU results and ≥1 audited
  DLA path.
- INT8-QAT loses ≤~1.0-1.5 mIoU vs FP32 per key subnet.
- At equal quality/risk, the method cuts p95 latency **or** J/frame by ≥20% on ≥3 of 4
  devices vs the matched baseline.
- Calibrated router meaningfully improves AURC/UIoU or risk-at-coverage, not just
  compute savings.
- Router + switching overhead <~5% of end-to-end cost.
- Results hold in sustained runs, not just short bursts before throttling.

**No-go / scope change**
- Hailo can't compile the decoder/resize after two weeks → shrink search space or run
  encoder-on-Hailo + host postprocess, and disclose the partition.
- Excessive DLA fallback → demote DLA to a secondary ablation; keep TensorRT GPU + Hailo
  as the two primary backends.
- Calibration doesn't improve external shift → drop "reliable", reframe as
  budget-aware deployment; never use "safety" language.
- Supernet loses >2 mIoU vs independent training at most budgets → reduce elasticity
  axes to width + resolution only.
- Speedup only comes from lower resolution → contribution is too weak; hardware-aware
  selection or calibrated routing must add independent benefit.

## 12. Manuscript structure (target)

Introduction (adverse vision + heterogeneous accelerators, state the four gaps directly,
no "deep learning has developed rapidly" opener) → Related work → Method
(search space, supernet training, measured-cost surrogate, calibrated router) →
Deployment protocol (export/compile, engine audit, measurement boundary,
reproducibility) → Experiments → Ablations and failure analysis → Limitations
(legacy Nano/Xavier stacks; calibration is not a safety certification; results are
compiler-version-dependent) → Conclusion.

Safe contribution phrasing for the first draft:

> We study the underexplored intersection of compiler-portable elastic segmentation,
> hardware-measured resource optimization, and calibrated risk-aware inference under
> adverse visual conditions.

Only upgrade to "to the best of our knowledge" after a systematic literature comparison
table is complete.

## 13. Timeline anchor

Primary target: IMAVIS special issue **Complex Environment Vision**, deadline
2027-02-15. See `docs/SOURCE_RESEARCH_GAP_2026.md` §10 for the full week-by-week
roadmap (weeks 1-2 hardware/toolchain inventory through weeks 23-24 submission
polish) — that section is reused as-is; update phase status here in `README.md`
instead of duplicating the table.

VP-NAV special issue (deadline 2026-12-31) is a stretch target only if a working
four-device baseline exists by mid-September 2026 and the method is stable by end of
October 2026; otherwise target the February deadline.

## 14. Locked decisions (source doc §13)

| Decision | Value | Status |
|---|---|---|
| End application | Road-scene adverse segmentation (default, autonomous-navigation framing) | locked as default; revisit only if a different end application is explicitly requested |
| GPU training capacity | `SERVER-01..05` per `../../docs/SHARED_INFRASTRUCTURE.md` §2 | locked to existing shared infra; no dedicated allocation yet |
| Exact Hailo variant | **Hailo-8, 26 TOPS** (not Hailo-8L) | **resolved 2026-09-08** via `hailortcli fw-control identify` on device `E1` — HailoRT 4.23.0, firmware 4.23.0, `hailo-all` 5.1.1 |
| Power meter / camera domain | Unknown — no external power analyzer or camera domain confirmed in shared infra | **open** — must be confirmed before any energy/camera claim |
| Device lineup (Nano vs new E4) | **E2 (Xavier NX) and E3 (AGX Xavier) both confirmed reachable with working TensorRT toolchains, 2026-09-08.** A **new device E4 (Thundercomm RUBIK Pi 3, Qualcomm QCM6490, Hexagon DSP/NPU via QAIRT)** was also granted access and is not part of the original 3-backend-class plan. | **E4: provisional call — kept secondary, not a required backend** (toolchain risk vs. deadline; see `INFRA_OVERRIDE.md`). Jetson Nano still unconfirmed. |
| Xavier DLA coverage | Compiler smoke test (2026-09-08, E2+E3) found DLA compiles the encoder but the decoder falls back to GPU due to a **confirmed hardware limit: 16 subgraphs per DLA core**, not an unsupported op. | **open** — either restructure the decoder to fit the budget, or scope DLA claims to the encoder only; see `docs/INFRA_OVERRIDE.md` and `reports/edge/` |

The energy/camera and device-lineup rows block Phase 1-2 experiments (not the supernet
code itself) and must be resolved before any Hailo/Qualcomm-specific number or a 4-vs-3
backend framing is locked into a manuscript.
