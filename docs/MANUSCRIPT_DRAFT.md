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
we select by measured deployment cost. For each of four deployment targets we
benchmarked — a Hailo-8 M.2 accelerator (Raspberry Pi 5 host) and three NVIDIA
Jetson boards spanning three tiers (Xavier NX, AGX Xavier, Orin Nano; TensorRT,
FP16) — we measure end-to-end and kernel-only latency for all four levels,
pooling three independent runs per level per the protocol in Section 4, and
build a per-target Pareto frontier over the (latency, mIoU) plane. On every
target, all four levels are Pareto-optimal (no level is simultaneously slower
and less accurate than another); the level that is optimal under a fixed 10 ms
latency budget differs by target — *small* on the Hailo-8, *small* on Xavier NX,
*medium* on Orin Nano, *large* on AGX Xavier, whose TensorRT GPU path is the
fastest of the four by a wide margin. Four real devices producing three
different choices at one identical budget is concrete evidence that
hardware-aware subnet selection is not interchangeable with a single,
device-agnostic choice — the premise our approach is built on — though we have
not yet quantified this against a FLOPs-aware selection baseline (Section 6
discusses this gap directly rather than treating the qualitative result as
sufficient on its own).

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
random initialization. Across three representative fixed-budget architectures
— Fast-SCNN (1.14 M parameters), DDRNet-23-slim (5.22 M parameters), and
SegFormer-B0 (3.72 M parameters) — and 4 seeds total, INT8 fine-tuning loses
between 0.08 and 1.25 mIoU points relative to the FP32 checkpoint it was
fine-tuned from, across Cityscapes and all four ACDC conditions (worst case:
Fast-SCNN seed 0 on rain, −1.25 points; SegFormer-B0 loses the least of any
architecture tested, −0.39 at worst, plausibly because our QAT mechanism only
fake-quantizes convolutional layers and SegFormer's encoder relies more on
non-convolutional attention/MLP-mixing operations). This is within the
go/no-go accuracy budget we fixed in advance (≤1.0–1.5 mIoU points; Section
7), confirmed across three architectures and four seeds total — **but only for
independent, fixed-architecture baselines.**

Applying the same recipe to the shared supernet's *large* level itself, across
three seeds, **does not reliably meet the same budget**: worst-per-seed loss is
−1.66, −1.39, and −2.54 points (mean −1.86, above the budget's upper bound),
with two of the three seeds exceeding it on at least one split. This is a
real, disclosed limitation, not an open question we expect to resolve in our
favor: quantization-aware training is reliable on independently-trained,
fixed-size architectures, but not (yet) on the shared elastic supernet as
currently trained and quantized. A plausible, untested explanation is that the
supernet's `large`-level weights are also exercised by every other elasticity
level through the sandwich rule, making them a harder quantization target than
a weight tensor serving a single fixed-size model alone; whether calibrated
(rather than dynamic) quantization ranges close this gap is the most promising
untested next step.

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

## 5. Experiments

### 5.1 Vision quality is monotonic in elasticity level

We first confirm the basic premise the elastic supernet depends on: that its four
trained levels form a genuine accuracy ladder, not four arbitrarily-ordered
configurations. At 100,000 training steps, mIoU is monotonic in model size at
every one of five evaluation splits (Table 1), and reproduces within 0.01–0.03
mIoU on an independently seeded run.

*Table 1. Supernet mIoU by elasticity level (seed 0, 100k steps).*

| Level | Cityscapes | ACDC/fog | ACDC/night | ACDC/rain | ACDC/snow |
|---|---:|---:|---:|---:|---:|
| tiny | 0.2986 | 0.3154 | 0.1944 | 0.2901 | 0.2610 |
| small | 0.3564 | 0.3658 | 0.2511 | 0.3680 | 0.3304 |
| medium | 0.4120 | 0.4360 | 0.2898 | 0.4003 | 0.3981 |
| large | 0.4712 | 0.4919 | 0.3312 | 0.4517 | 0.4492 |

ACDC/night is the hardest condition throughout, as expected of the lowest-light,
lowest-contrast adverse condition. That ACDC/fog scores *above* clean Cityscapes
at every level is a real effect we traced to class composition rather than an
artifact of unweighted macro-averaging: several large structural classes (wall,
pole, traffic light, sky) score markedly higher in fog's visually simpler scenes,
while dynamic road-user classes degrade as expected under fog (at the *large*
level: person 0.471→0.280, bicycle 0.472→0.204, car 0.828→0.709). We report
per-class breakdowns for exactly these dynamic classes alongside every aggregate
mIoU number in this paper for this reason — an aggregate number alone would mask
a real safety-relevant degradation behind an unrelated class-mix effect.

### 5.2 Shared supernet vs. independent per-budget training

RQ2 asks whether a single shared supernet can match independently trained,
same-parameter-budget models without the cost of training and maintaining one
model per target — not whether it can beat them. We compare the supernet's
*large* level (1.02 M parameters) against Fast-SCNN (1.05 M parameters, the one
comparison baseline at a matched budget), both trained with identical
augmentation, over three independent seeds per side, per our go/no-go criterion
that headline accuracy numbers require three training seeds.

*Table 2. Supernet-large vs. Fast-SCNN, augmented, 3-seed average.*

| Split | Fast-SCNN (aug) | Supernet-large (aug) | Gap |
|---|---:|---:|---:|
| Cityscapes | 0.5228 | 0.5338 | +0.0110 |
| ACDC/fog | 0.5629 | 0.5650 | +0.0021 |
| ACDC/night | 0.3822 | 0.3757 | −0.0065 |
| ACDC/rain | 0.5050 | 0.5027 | −0.0023 |
| ACDC/snow | 0.5048 | 0.5055 | +0.0007 |

The supernet wins three of five splits and loses two, with every margin within
1.1 mIoU points — a near-exact tie on ACDC/snow. We read this as **true
near-parity, not a win for either side**: the shared supernet matches
independently trained, same-budget training within roughly one mIoU point
across clean and all four adverse conditions, while requiring one training run
instead of four (or four times four, across our elasticity levels). This is
RQ2's actual hypothesis, confirmed rather than exceeded, and is comfortably
clear of our own no-go trigger (>2 mIoU loss in either direction). We note this
result required three successive corrections during development — an
un-augmented, single-seed comparison had initially favored the supernet by
2.6–6.4 mIoU, an artifact of Fast-SCNN's own augmentation gap rather than a real
capability difference — and report only the final, 3-seed, augmented-vs-augmented
number as citable.

### 5.3 Data augmentation's effect is architecture-dependent, not monotonic in model size

Motivated by an unexplained result on one architecture during the comparison
above, we trained all five of our required baseline architectures both with and
without our augmentation policy (random scale-crop, horizontal flip, color
jitter), spanning almost an order of magnitude in parameter count (1.14 M–11.0 M).
The effect is not a monotonic function of size (Table 3).

*Table 3. Augmentation effect (augmented − non-augmented mIoU) by architecture.*

| Model | Params (M) | No-aug Cityscapes mIoU | Cityscapes Δ | ACDC-average Δ |
|---|---:|---:|---:|---:|
| Fast-SCNN | 1.14 | 0.408 | **+0.115** | **+0.107** |
| BiSeNetV2 | 1.71 | 0.581 | −0.004 | +0.003 |
| SegFormer-B0 | 3.72 | 0.567 | −0.015 | +0.016 |
| DDRNet-23-slim | 5.22 | 0.627 | +0.000 | −0.005 |
| MobileNetV3+DeepLabV3 | 11.03 | 0.577 | **+0.026** | **+0.045** |

The smallest model (Fast-SCNN) is, without augmentation, badly underfit — its
0.408 Cityscapes mIoU is far the worst of the five — and augmentation relieves
this directly, producing the largest gain of any architecture. The *largest*
model (MobileNetV3+DeepLabV3) also gains substantially, despite showing no sign
of underfitting: its non-augmented mIoU (0.577) is in fact *worse* than
DDRNet-23-slim's (0.627), a model with under half its parameters, a pattern
consistent with overfitting our comparatively small (4,575-image) training set —
a failure mode augmentation's regularizing effect also relieves, through a
different mechanism than Fast-SCNN's. The three architectures between these
extremes (1.7–5.2 M parameters, the three best-fitting models without
augmentation) show small, mixed, and sometimes slightly negative effects. We
read this as a bimodal relationship with how comfortably a given architecture
fits this particular dataset and training budget, not a monotonic function of
parameter count — and caution against citing "augmentation helps small models
more" as a general claim without first establishing whether a given architecture
is under- or over-fit at its evaluated budget.

### 5.4 Measured latency and hardware-aware subnet selection

*Table 4. End-to-end latency by elasticity level, mean over 3 independent runs,
across all four benchmarked deployment targets.*

| Level | Hailo-8 (ms) | Xavier NX, TensorRT FP16 (ms) | AGX Xavier, TensorRT FP16 (ms) | Orin Nano, TensorRT FP16 (ms) |
|---|---:|---:|---:|---:|
| tiny | 3.714 | 2.061 | 0.912 | 1.081 |
| small | 6.597 | 4.964 | 1.781 | 2.604 |
| medium | 20.29 | 15.10 | 4.546 | 7.727 |
| large | 41.14 | 40.26 | 9.389 | 17.95 |

Joining this table with Table 1's mIoU numbers per level (Section 3.2), all four
levels lie on the Pareto frontier on every target — none is simultaneously
slower and less accurate than another — and the level selected under a fixed
10 ms budget differs by target: *small* on the Hailo-8, *small* on Xavier NX,
*large* on AGX Xavier (whose TensorRT path is the fastest of the four, 4–4.5×
the Hailo-8's per-level latency), and *medium* on Orin Nano. Four real devices
producing three distinct choices at one identical latency budget is the full
device-lineup evidence our protocol calls for — stronger than a two-device
demonstration alone, though still only a qualitative confirmation that
hardware-aware selection matters, not yet the quantitative comparison against a
FLOPs-aware baseline that would let us claim a specific margin (Section 6).

## 6. Ablations

*(Preliminary — one axis of ten, one seed per side. Included here as an honest
first data point, not a settled result; do not cite the pattern below without
a repeat run.)*

We compare the supernet trained with our default in-place distillation weight
against an otherwise-identical run with distillation disabled
(`search.alpha_distill = 0`), both at 100k steps, evaluated at all four
elasticity levels (Table 5).

*Table 5. Distillation effect (no-distillation − with-distillation mIoU,
points) by elasticity level.*

| Level | Cityscapes | ACDC/fog | ACDC/night | ACDC/rain | ACDC/snow | Mean |
|---|---:|---:|---:|---:|---:|---:|
| tiny | +0.53 | −0.27 | +1.04 | +0.02 | +0.16 | +0.30 |
| small | −0.47 | −0.36 | −0.40 | −1.35 | −0.74 | **−0.66** |
| medium | −0.69 | +0.68 | −0.22 | +0.48 | +1.12 | +0.27 |
| large | +1.33 | +1.08 | −0.32 | −0.55 | +1.67 | +0.64 |

Distillation's benefit is not uniform across levels: at the *small* level
every split is worse without it, the only fully one-directional result in the
table; at *tiny*, *medium*, and *large* the effect is mixed and, in this
single-seed comparison, slightly favors *not* using distillation. This runs
against the simple expectation that distillation from the *large* teacher
level should help every smaller level comparably. We flag this as a real,
if preliminary, finding rather than noise we assume away — but this project
has already learned once that a single-seed comparison can flip direction
(Section 5.2's original supernet-vs-baseline result), so we withhold a
mechanistic claim until a second seed pair is run.

The remaining nine ablation axes (FLOPs- vs. measured-latency objective,
latency-only vs. latency+energy, unconstrained vs. compiler-safe operator
space, PTQ vs. QAT, static routing vs. router — already covered in Section
3.4's 7/7 result — entropy vs. calibrated-risk routing — also Section 3.4 —
per-frame vs. temporal-window routing, and behavior across power modes) are
either untested or blocked on infrastructure we do not yet have (a calibrated
power meter, principally).

---

## Traceability (remove before submission; keep while drafting)

Every quantitative claim above traces to a specific report, listed here so a
later editing pass can verify nothing drifted from its source during rewriting:

- §3.1 dataset/split sizes, augmentation: `README.md` Phase 4;
  `reports/augmentation_effect_all_5_baselines_20260916.md`.
- §3.2 Pareto frontier, 10 ms budget example, 4–4.5× TensorRT/Hailo ratio:
  `reports/pareto_search_v1_20260912.md`.
- §3.3 QAT recipe, fine-tune steps/LR, mIoU-loss range, go/no-go pass/supernet
  gap: `reports/qat_v1_20260913.md` (2026-09-14/15/16/17 updates).
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
- §5.1 monotonicity table, fog/class-mix finding, per-class numbers:
  `reports/first_full_supernet_run_100k_20260910.md`.
- §5.2 go/no-go 3-seed table, correction history: README.md Phase 8;
  `reports/baseline_comparison_gap_check_20260912.md`.
- §5.3 5-architecture augmentation table:
  `reports/augmentation_effect_all_5_baselines_20260916.md`.
- §3.2/§5.4 4-device latency table, 3-way Pareto selection split at 10ms:
  `reports/pareto_search_v1_20260912.md` (E1/E3 original);
  `reports/edge/E2_E5_tensorrt_benchmark_all_levels_20260916.md` (E2/E5
  extension, same day).
- §6 distillation ablation table: `reports/ablation_distillation_v1_20260917.md`.

## Not yet draftable (do not backfill without new results)

- §1 Introduction / §2 Related work: blocked on the systematic literature
  review (`RESEARCH_PLAN.md` §2), not started.
- §6 Ablations: drafted with 1 of 10 axes (distillation), explicitly
  preliminary (n=1 per side) -- 9 axes remain untested or infrastructure-
  blocked. Expand as more axes get real results; do not overwrite the
  preliminary framing until a second seed pair confirms the distillation
  pattern.
- §7 Limitations / §8 Conclusion: blocked on Ablations reaching a more
  complete state, and on Introduction/Related work above.
