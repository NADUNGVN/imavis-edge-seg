# PACE-Seg: Platform-Aware Calibrated Elastic Semantic Segmentation for Reliable Edge Vision

> **Draft manuscript, section by section.** Distinct from `docs/MANUSCRIPT_SKELETON.md`
> (the living status tracker, annotated with internal report citations and
> `[TODO: ...]` markers) — this file is the actual paper text, written in submission
> voice. A section is added here only once its skeleton counterpart has real,
> committed results behind it; a section's absence here means it is not ready to
> draft yet, not that it was skipped. Started 2026-09-16 with Method and Deployment
> Protocol (the two sections with the most complete real evidence). Numbers cited
> here must trace to a specific `reports/*.md` file — if a number in this draft
> cannot be traced to a report, that is a bug in the draft, not a shortcut to take.

---

## 3. Method

### 3.1 Compiler-safe elastic supernet

We train a single shared segmentation network that executes at four discrete
elasticity levels — *tiny*, *small*, *medium*, *large* — obtained by varying
channel width multiplier, block count, and input resolution. Every operator in
the network (convolution–batch-norm–activation, depthwise/pointwise convolution,
pooling, elementwise add/concatenate, static-factor bilinear resize) is drawn
from the intersection of operators we verified compile and execute correctly on
all three target compiler backends (Section 3.4), rather than from an
unconstrained search space later found to be unsupported on one target. All four
levels are trained jointly with a sandwich-rule schedule (training the smallest,
largest, and a randomly sampled subset of middle levels at each step) and
in-place distillation from the largest level's output to the smaller levels,
with a boundary-aware auxiliary loss term that up-weights pixels near class
boundaries — the region where small classes are disproportionately lost at low
capacity.

Training uses real Cityscapes (2,975 train / 500 val) and ACDC (1,600 train /
406 val, four adverse conditions: fog, night, rain, snow) images at a batch size
and learning rate schedule held fixed across every experiment in this paper. We
additionally apply random scale-crop-pad, horizontal flip, and color jitter
augmentation to the training stream; Section 5 quantifies this choice's effect
across five different segmentation architectures rather than assuming it helps
uniformly.

### 3.2 Hardware-in-the-loop Pareto subnet selection

Rather than selecting among the four trained levels by parameter count or FLOPs,
we select by measured deployment cost. For each of two deployment targets we
benchmarked — a Hailo-8 M.2 accelerator (Raspberry Pi 5 host) and an NVIDIA
Jetson AGX Xavier (TensorRT, FP16) — we measure end-to-end and kernel-only
latency for all four levels, pooling three independent runs per level per the
protocol in Section 4, and build a per-target Pareto frontier over the
(latency, mIoU) plane. On both targets, all four levels are Pareto-optimal (no
level is simultaneously slower and less accurate than another); the level that
is optimal under a fixed latency budget differs by target — the *small* level is
preferred on the Hailo-8 at a 10 ms budget, while the same budget selects the
*large* level on the AGX Xavier, whose TensorRT GPU path is roughly 4–4.5×
faster per level. This is concrete evidence that hardware-aware subnet selection
is not interchangeable with a single, device-agnostic choice — the premise our
approach is built on — though we have not yet quantified this against a
FLOPs-aware selection baseline (Section 6 discusses this gap directly rather
than treating the qualitative result as sufficient on its own).

### 3.3 Quantization-aware training

We convert every convolution — both the plain convolutions in our comparison
baselines and the width-sliced `SlimmableConv2d` layers specific to the shared
supernet — to a fake-quantized form: dynamic, per-tensor, symmetric INT8
quantization of each layer's weight and input activation, applied with a
straight-through gradient estimator so training proceeds unmodified through the
quantization op. For the supernet's slimmable convolutions, the quantization
range is computed from each elasticity level's own *active* channel slice at
every forward call, so every level receives an appropriately scaled range
without any additional bookkeeping. We deliberately use a dynamic, uncalibrated
range rather than a range fit from a held-out calibration set spanning all five
visual conditions (Section 3.1) — a documented simplification that should be
read as an upper bound on achievable accuracy, not a final number, until a
calibrated variant is built.

We fine-tune each INT8 model for 10,000 steps from a converged FP32 checkpoint
at a reduced learning rate ($3\times10^{-5}$), rather than training INT8 from
random initialization. Across two representative fixed-budget architectures —
Fast-SCNN (1.14 M parameters) and DDRNet-23-slim (5.22 M parameters) — and two
random seeds each, INT8 fine-tuning loses between 0.11 and 1.25 mIoU points
relative to the FP32 checkpoint it was fine-tuned from, across Cityscapes and
all four ACDC conditions (worst case: Fast-SCNN seed 0 on rain, −1.25 points).
This is within the go/no-go accuracy budget we fixed in advance (≤1.0–1.5 mIoU
points; Section 7), confirmed across two architectures and two seeds rather than
a single run.

### 3.4 Compiled static engines and calibrated routing

Each trained elasticity level is exported and compiled independently — never as
a single graph with runtime branching — to a TensorRT engine, a DLA-targeted
engine, and a Hailo HEF (via ONNX and the Hailo Dataflow Compiler); a deployed
system picks one compiled engine per inference window rather than executing
dynamic control flow inside a compiled graph. This partition into
independently-compiled static engines is what makes hardware-aware routing
possible without a graph any of the three compiler toolchains would reject.

The routing decision is made by a lightweight, calibrated risk probe rather than
a fixed policy. We run one cheap forward pass — typically at the *tiny* level —
and take the mean per-pixel softmax entropy of its output as a raw,
uncalibrated risk score; we then map this score to an expected pixel-error
probability using a monotonic calibrator (quantile-binned with a cumulative-max
correction) fit on a held-out half of each validation split, never on the images
a policy is later evaluated against. At inference time, a target risk budget
determines how many elasticity levels to escalate above the cheapest available
level.

We evaluate this policy end to end: for the held-out half of each validation
split, we run every candidate level's forward pass, route each image under each
candidate policy using only its cheap-pass risk score, and measure the resulting
*achieved* mIoU and the real measured latency (Section 3.2's hardware table) the
policy actually spends. Calibrated routing achieves higher mIoU than routing on
raw, uncalibrated entropy in every one of seven tested conditions — Cityscapes
under two risk-budget settings, two deployment targets (Hailo-8 and AGX Xavier),
and all four ACDC adverse conditions — with margins of +0.0013 to +0.0183 mIoU.
The efficiency picture is more mixed: because calibrated routing also spends
slightly more latency than entropy-only routing in every condition, the
mIoU-gained-per-millisecond-spent comparison favors calibrated routing in two of
the seven conditions and entropy-only routing in the other five. We report both
numbers rather than only the mIoU comparison that favors our method.

A structural property of the current policy is worth stating plainly: escalation
decisions depend only on each candidate level's latency *rank*, not its
magnitude, so a calibrator fit once transfers unchanged across deployment
targets — confirmed empirically (Hailo-8 and AGX Xavier produce numerically
identical achieved mIoU on the same validation split, despite a >4× difference
in absolute per-level latency). This is a useful robustness property — no
per-device recalibration is required — but it also means the current policy
cannot yet exploit a target-specific cost *ratio* between levels, which we flag
as a direction for a latency-value-aware successor policy rather than treating
as resolved.

---

## 4. Deployment protocol

### 4.1 Compiler validation

We validate each of the three target compiler toolchains independently before
relying on any of them for a result: TensorRT (GPU and DLA paths, via `trtexec`),
and the Hailo Dataflow Compiler (ONNX → HAR → HEF). TensorRT compiles and
executes all four elasticity levels in both FP16 and INT8 precision on two
Jetson devices (8/8 configurations pass); the Hailo toolchain compiles all four
levels to a working HEF, and all four ran successfully on physical Hailo-8
hardware. Xavier's DLA path compiles only the encoder half of the network at
every elasticity level — the decoder always falls back to the GPU. We traced
this to a hard, documented hardware limit (a DLA core executes at most 16
subgraphs, a budget the encoder alone exhausts), not an unsupported operator,
ruling out a fix by further restricting our operator set. Because this
contingency was anticipated in our go/no-go criteria (excessive DLA fallback),
we demote DLA to a secondary, encoder-only ablation and report TensorRT GPU and
Hailo HEF as the two primary deployment backends for every headline result in
this paper.

### 4.2 Latency measurement protocol

Latency is measured with a protocol adapted from MLPerf Power's methodology
(without claiming formal compliance): batch size 1; identical resolution and
pre/post-processing held fixed within a configuration; a warm-up period before
timed measurement; three independent runs per (device, backend, precision,
elasticity level) configuration, pooled into one bootstrap 95% confidence
interval rather than reported as three separate point estimates; kernel-only
and end-to-end latency reported separately, never compared across device
families where the two are not measuring the same boundary. We report the
complete measured cross-backend latency table for the Hailo-8 and AGX Xavier
targets across all four elasticity levels in Section 5.

**Energy is not yet reported.** The Hailo-8 M.2 module used in this work
exposes no on-board power or current telemetry (`--measure-power` and
`--measure-current` are both unsupported on this hardware), and we do not yet
have access to an external calibrated power meter for either target — a
material gap against our own protocol (item 6 above), disclosed rather than
worked around with an unaudited on-chip estimate.

---

## Traceability (remove before submission; keep while drafting)

Every quantitative claim above traces to a specific report, listed here so a
later editing pass can verify nothing drifted from its source during rewriting:

- §3.1 dataset/split sizes, augmentation: `README.md` Phase 4;
  `reports/augmentation_effect_all_5_baselines_20260916.md`.
- §3.2 Pareto frontier, 10 ms budget example, 4–4.5× TensorRT/Hailo ratio:
  `reports/pareto_search_v1_20260912.md`.
- §3.3 QAT recipe, fine-tune steps/LR, mIoU-loss range, go/no-go pass:
  `reports/qat_v1_20260913.md` (2026-09-14/15/16 updates).
- §3.4 compiled-engine partition, DLA 16-subgraph limit (referenced, detailed in
  §4): `README.md` Phase 2; `reports/edge/E3_compiler_smoke_test_20260908.md`.
- §3.4 router calibration/evaluation protocol, 7/7 result, efficiency split,
  latency-rank-only structural finding: `reports/router_v1_20260914.md`.
- §4.1 compiler validation counts (8/8 TensorRT, 4/4 Hailo compile+hardware,
  DLA 16-subgraph limit, DLA demotion decision): `README.md` Phase 2;
  `reports/edge/E3_compiler_smoke_test_20260908.md`;
  `reports/edge/E1_hailo_dfc_compile_20260909.md`.
- §4.2 latency protocol design, bootstrap CI, kernel-only vs. end-to-end
  separation, Hailo-8 telemetry gap: `README.md` Phase 3;
  `reports/edge/E1_hailo_benchmark_protocol_20260910.md`;
  `reports/edge/E3_tensorrt_benchmark_all_levels_20260911.md`;
  `RESEARCH_PLAN.md` §9.

## Not yet draftable (do not backfill without new results)

- §1 Introduction / §2 Related work: blocked on the systematic literature
  review (`RESEARCH_PLAN.md` §2), not started.
- §5 Experiments: partially draftable (the go/no-go comparison and the 5-model
  augmentation finding have real numbers) but not started here yet.
- §6 Ablations and failure analysis: blocked on Phase 9, ~10% started (only
  2-3 of 10 axes touched as side effects of other work).
- §7 Limitations / §8 Conclusion: blocked on the above.
