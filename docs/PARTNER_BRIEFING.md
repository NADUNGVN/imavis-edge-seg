# Partner briefing — PACE-Seg (this repo) status + how to pick a non-overlapping next branch

> For a collaborator opening a new research branch/track that shares this repo's
> hardware fleet and infrastructure. Read this before scoping a new topic so it
> complements PACE-Seg instead of duplicating it.

## 1. What PACE-Seg is

**PACE-Seg** (Platform-Aware Calibrated Elastic Semantic Segmentation for Reliable
Edge Vision): one elastic **segmentation** supernet, trained once, exported as 3-4
compiler-compatible static engines per accelerator backend (TensorRT GPU, Hailo
dataflow NPU; Xavier DLA demoted to a secondary ablation, see §3). A calibrated
visual-risk router will eventually pick an engine per frame window under a measured
latency/energy budget, not FLOPs. Verified on real hardware (Jetson AGX/NX, Pi 5 +
Hailo-8) under clean/night/rain/fog/snow (Cityscapes + ACDC). Target: *Image and
Vision Computing* (IMAVIS), Elsevier, special issue "Complex Environment Vision",
deadline 2027-02-15. Full design in `docs/RESEARCH_PLAN.md`.

**Task = semantic segmentation.** This matters for scoping a new branch (§4) --
several adjacent edge-AI research gaps are framed around *object detection*, a
different task, which is the cleanest way to avoid overlap.

## 2. What's done (as of 2026-09-12)

- **Hardware/toolchain inventory**: E1 (Pi5+Hailo-8), E2 (Xavier NX), E3 (AGX
  Xavier), E5 (Orin Nano Super) all have working, verified ML toolchains. 5 train
  servers (SERVER-01..05) inventoried.
- **Compiler smoke test, all 3 backend families PASS** (TensorRT GPU, Xavier DLA,
  Hailo DFC) — compile *and* real-hardware execution, all 4 elasticity levels.
  **Xavier DLA demoted to a secondary/encoder-only ablation** (confirmed hardware
  limit: 16-subgraph-per-core budget, exhausted by the encoder alone) — TensorRT GPU +
  Hailo are now the two primary backends. See `reports/edge/`.
- **Real data pipeline**: Cityscapes (2975 train/500 val) + ACDC (1600 train/406 val,
  4 adverse conditions), correct trainId mapping, real directory layout.
- **Elastic supernet** (`src/imavis_edge_seg/models/`): 4 elasticity levels
  (tiny/small/medium/large), sandwich-rule training + in-place distillation +
  boundary-aware loss, ~1.05M total params (shared across levels).
- **Latency benchmark harness** (`src/imavis_edge_seg/benchmark/`): MLPerf-Power-
  inspired protocol, real data on 2 backends x 4 levels (E1 Hailo, E3 TensorRT GPU) --
  `outputs/benchmark_lookup_table.csv`. **No external power meter yet** (deferred by
  choice) — energy/frame claims are on hold, latency-only for now.
- **2/3 supernet training seeds complete** (100k steps each, real GPU time,
  `pace_seg_v1` seed 0 + seed 1), consistent to ~0.01-0.03 mIoU.
- **5/7 required baselines implemented, trained and evaluated** from scratch
  (`fast_scnn`, `bisenetv2`, `ddrnet23_slim`, `segformer_b0` -- own implementations;
  `mobilenetv3_deeplabv3` via torchvision). `hard`/`ucpnet` skipped -- no public
  code/checkpoint found (per the plan's own conditional inclusion rule).
- **First real go/no-go signal, positive**: `fast_scnn` (1.136M params, the one
  same-parameter-budget baseline) loses to the supernet's "large" level on **every**
  split (Cityscapes + 4 ACDC conditions), by 2.6-6.4 mIoU points --
  `reports/baseline_comparison_gap_check_20260912.md`.
- **Training-time data augmentation** just added (random scale+crop, flip, color
  jitter) -- none of the results above used it yet.

## 3. Locked/resolved decisions relevant to scoping a new branch

- DLA is secondary, not a required backend (§2 above).
- Energy/J-frame claims are deferred pending a power meter purchase (a Shelly Plus
  Plug S was recommended, not yet bought).
- Device lineup: E4 (RUBIK Pi 3, Qualcomm) is opportunistic/secondary, not required.
  E5 (Orin Nano Super) stands in for the legacy "Jetson Nano" slot until/unless a
  legacy board is provided.

## 4. What PACE-Seg will do next (so a new branch doesn't duplicate it)

In rough priority order: finish `segformer_b0` + a 3rd supernet seed (in progress);
build the **Pareto search** algorithm (Contribution 2, consumes the latency lookup
table above); build the **calibrated visual-risk router** (Contribution 3); **QAT**
(quantization-aware training) adapted to the shared-weight elastic architecture
(Contribution 6, INT8 go/no-go target: lose ≤1.0-1.5 mIoU vs FP32); full multi-seed
experiments across all baselines/conditions; ablations (§10: shared vs. independent
training, FLOPs vs. measured-cost objective, PTQ vs. QAT, with/without distillation,
entropy vs. calibrated router, per-frame vs. windowed routing, power-mode sweep);
manuscript.

## 5. Candidate next branches and their overlap risk with PACE-Seg

A separate deep-research doc (`deep_research_gaps_3_edge_ai_topics_public_data.docx`,
outside this repo) scopes 3 **object-detection** edge-AI topics, all usable with this
same hardware fleet. Task difference (detection vs. segmentation) alone doesn't
guarantee no overlap -- check the *method* and *domain* too:

| Candidate | Method overlap with PACE-Seg | Domain overlap | Overlap risk |
|---|---|---|---|
| **#1 HeteroAdapt-Vision** (SLA-aware cross-accelerator controller, chooses model/resolution/precision per frame) | **High** -- this is conceptually the same problem as PACE-Seg's own Contribution 2 (hardware-in-the-loop Pareto selection) and Contribution 3 (calibrated router), just for detection instead of segmentation | Low (no adverse-weather framing required) | **Medium-high** -- risk of two papers making the same systems argument on two tasks; needs explicit differentiation if chosen (e.g. focus on the *action-space portability* problem specifically, which PACE-Seg doesn't address since it only targets 2 backends, not 3 heterogeneous vendor ecosystems) |
| **#2 CalibShift-INT8** (condition-aware INT8 calibration profiles for adverse weather) | **Medium** -- adjacent to PACE-Seg's still-unstarted QAT work (Contribution 6) and both use ACDC | **High** -- same dataset (ACDC), same "adverse weather robustness" framing | **Medium** -- lowest engineering risk per the source doc, but the closest in spirit to what PACE-Seg will eventually do for quantization; would need a clearly distinct angle (e.g. PTQ calibration-bank selection vs. PACE-Seg's QAT-on-a-shared-elastic-backbone) to avoid the two projects reading as the same idea |
| **#3 SplitRoute-Det** (edge-cloud collaborative detection: dynamic partitioning + early exit + feature compression) | **Low** -- a genuinely different systems concern (network split point / bandwidth-aware partitioning), nothing in PACE-Seg addresses edge-cloud collaboration at all | Low (no required adverse-weather angle) | **Low -- cleanest choice for non-overlap** |

**Recommendation if picking from that document**: **#3 SplitRoute-Det** has the least
conceptual overlap with PACE-Seg and would read as clearly complementary work (this
repo: on-device elastic model + hardware-aware routing; a new branch: edge-cloud
partitioning) even though both branches would share the same physical device fleet
and could reuse this repo's benchmark-harness code (`src/imavis_edge_seg/benchmark/`)
as a starting point rather than rebuilding latency measurement from scratch. The
source doc itself rates #3 as the highest-*risk* to execute (compiler/runtime work
across 3 vendor stacks) -- that's an execution-difficulty tradeoff against overlap
risk, worth discussing with whoever scopes the new branch before committing.

If the partner instead prefers #1 or #2 for lower execution risk, flag explicitly in
that branch's own research plan how it differs in contribution from PACE-Seg's
existing/planned Pareto-search, router, and QAT work, so the two don't compete for
the same claim later.
