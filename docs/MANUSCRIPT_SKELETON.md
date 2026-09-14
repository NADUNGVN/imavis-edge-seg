# PACE-Seg — manuscript skeleton

> **Living skeleton, not a draft manuscript.** Snapshot date: 2026-09-13. This file
> tracks what can honestly be written *today* against `docs/RESEARCH_PLAN.md` §12's
> target structure, grounded only in what `README.md`'s status table and the reports
> under `reports/`/`reports/edge/` have actually established. Regenerate/update this
> file as more results land — a later session should treat it as something to revise
> in place, not a frozen document. Do not delete `**[TODO: ...]**` markers by writing
> around them; resolve them only when the cited deliverable has actually landed.

---

## Introduction

Segmentation on heterogeneous edge accelerators under adverse visual conditions is
addressed only piecemeal by existing work: proxy cost metrics (FLOPs/params) do not
track measured latency or energy across accelerator architectures (G1); a graph
compiled for one accelerator is not guaranteed to compile, unmodified, for another
(G2); clean-image mIoU does not reflect reliability under night, rain, fog and snow
(G3); and one independently-trained model per device wastes training/maintenance
cost that a shared elastic architecture could avoid (G4). We state these four gaps
directly, not via a generic "deep learning has developed rapidly" opener.

> We study the underexplored intersection of compiler-portable elastic segmentation,
> hardware-measured resource optimization, and calibrated risk-aware inference under
> adverse visual conditions.

(Verbatim, `docs/RESEARCH_PLAN.md` §12 — deliberately not upgraded to "the first" or
"to the best of our knowledge," which requires a systematic literature search not
yet done, per §2/§12.) The four intended contributions (§4) are a compiler-safe
elastic segmentation supernet, a hardware-in-the-loop Pareto search over measured
cost, a calibrated visual-risk router, and a cross-platform reliability benchmark
protocol.

**[TODO: pending systematic literature review, RESEARCH_PLAN.md §2]** Any novelty
claim stronger than the safe phrasing above needs the Scopus/Web of Science search
§2 requires, close to submission — not started.

## Related work

Per §2's four gaps and `docs/SOURCE_RESEARCH_GAP_2026.md`'s survey, this section
needs: (1) lightweight/real-time segmentation reviews and the FLOPs-vs-latency
disconnect (G1); (2) compiler-portability work across TensorRT, DLA and Hailo-style
dataflow NPUs, including DLA's known static-shape/operator limits (G2); (3)
adverse-condition segmentation and uncertainty benchmarks, notably ACDC, and how
mean mIoU can mask class-level degradation (G3); (4) dynamic/once-for-all supernet
and routing literature as the state of the art for avoiding per-budget independent
training (G4). Adjacent named systems (YOLIC, Light-SEF, UCPNet, HARD) need
positioning against PACE-Seg's specific combination, not any single axis.

**[TODO: pending systematic literature review]** The actual deliverable — a method
vs. G1-G4 coverage table — cannot be written credibly from the informal source
survey alone.

## Method

The supernet (`src/imavis_edge_seg/models/`) uses a restricted operator set —
Conv-BN-act, depthwise/pointwise conv, pooling, add/concat, static resize — chosen to
sit in the intersection of what TensorRT, Xavier DLA and the Hailo Dataflow Compiler
can each execute, trained once across four elasticity levels (`tiny`/`small`/
`medium`/`large`, varying channel multiplier, block count and input resolution) via
a sandwich rule with in-place distillation and a boundary-aware auxiliary loss for
small classes at low capacity (§5.1). Rather than optimizing FLOPs, the objective
(§5.4) trades segmentation, distillation and calibration loss terms against
surrogates for measured p95 latency and energy, fit from real hardware
measurements.

The supernet architecture, sandwich-rule/distillation/boundary-aware training loop
and mIoU evaluation loop are complete (README.md Phase 4). The measured-cost
surrogate is realized today only as a discrete lookup-table Pareto search over the
four already-trained levels (`src/imavis_edge_seg/search/pareto.py`, Phase 5) — not
yet a continuous surrogate feeding back into architecture search, and latency-only
(no energy term).

**[TODO: pending a real QAT training run + FP32-vs-INT8 mIoU comparison, Phase 6]**
QAT v1 (`src/imavis_edge_seg/training/quantization.py`) exists as a mechanism only:
dynamic per-tensor INT8 fake-quantization convertible in place on a model while
preserving `state_dict` key names (enabling the §5.2 FP32→QAT workflow), verified
against the real `fast_scnn` architecture. Not yet done: calibrated (not dynamic)
quantization ranges; support for the supernet's `SlimmableConv2d`; and any actual
training run, so the §11 bar of ≤~1.0-1.5 mIoU loss vs. FP32 cannot be written up
yet. Compiler-side quantization so far is PTQ-only and uncalibrated (smoke tests used
`--int8` with no calibration data, or Hailo's `--use-random-calib-set`), a separate
and even less rigorous placeholder than QAT v1.

**[TODO: pending calibrated risk router, Phase 7]** The router (Contribution 3) is
unimplemented; only its calibration-metric prerequisite
(`src/imavis_edge_seg/evaluation/calibration.py`: ECE, NLL, Brier, AURC) exists. No
routing policy can be described yet.

## Deployment protocol

Each trained level is exported and compiled independently to a TensorRT engine
(GPU), a DLA-targeted engine (Xavier) and a Hailo HEF (ONNX → Dataflow Compiler),
with no dynamic control flow inside any compiled graph; a router would switch
engines on an 8-32 frame window to amortize switching cost (§5.3). Compiler
validation is real and complete across all three backend families
(`reports/edge/E3_compiler_smoke_test_20260908.md`,
`reports/edge/E1_hailo_dfc_compile_20260909.md`): TensorRT GPU compiles and runs all
four levels (8/8 FP16/INT8 PASS on E2/E3); Hailo's DFC compiles all four levels to a
working HEF, all of which ran successfully on real Hailo-8 hardware (E1); Xavier DLA
compiles only the encoder — the decoder always falls back to GPU because of a
confirmed hardware limit of 16 subgraphs per DLA core, already exhausted by the
encoder alone, not an unsupported-operator problem. Per §11's "excessive DLA
fallback" contingency, DLA was demoted (2026-09-10) to a secondary, encoder-only
ablation; TensorRT GPU and Hailo HEF are the two primary backends for headline
claims.

The latency protocol follows MLPerf Power's principles without a compliance claim
(§9): batch 1, ≥200-inference warm-up, ≥5,000 frames, three independent runs, full
toolchain disclosure. It has run live end to end on E1 (Hailo) and E3 (TensorRT
GPU), all 4 levels x 3 runs each; kernel-only and end-to-end latency are reported
and never conflated across backends. E1's Hailo-8 module lacks an on-board
power/current sensor, so energy/frame there requires an external meter and is
recorded as unavailable, never estimated from telemetry.

**[TODO: pending external power meter, RESEARCH_PLAN.md §14]** No energy/J-frame
number exists for any device; the default objective is latency-only until a meter
is acquired.

**[TODO: pending TensorRT benchmarking on E2/E5]** Only E3 has a full 4-level
TensorRT sweep; E2 and E5 have working toolchains but no benchmark run yet
(`reports/edge/E3_tensorrt_benchmark_all_levels_20260911.md`).

## Experiments

**Vision quality, 100k-step run, seed 0** (`reports/first_full_supernet_run_100k_20260910.md`):

| level | Cityscapes | ACDC/fog | ACDC/night | ACDC/rain | ACDC/snow |
|---|---:|---:|---:|---:|---:|
| tiny | 0.2986 | 0.3154 | 0.1944 | 0.2901 | 0.2610 |
| small | 0.3564 | 0.3658 | 0.2511 | 0.3680 | 0.3304 |
| medium | 0.4120 | 0.4360 | 0.2898 | 0.4003 | 0.3981 |
| large | 0.4712 | 0.4919 | 0.3312 | 0.4517 | 0.4492 |

mIoU is monotonic in model size at every condition, reproducing within 0.01-0.03
mIoU on an independent seed. ACDC/night is the hardest condition throughout, as
expected. ACDC/fog scoring above clean Cityscapes is a real, investigated finding,
not an artifact: several large, structural classes (wall, pole, traffic light, sky)
score markedly higher in fog's visually simpler scenes, while dynamic road-user
classes degrade as expected (person 0.471→0.280, bicycle 0.472→0.204, car
0.828→0.709 at `large`) — unweighted macro mIoU hides this, which is why per-class
breakdowns for person/rider/pole/sign/light are reported alongside the aggregate,
not as an optional extra.

**Go/no-go: supernet vs. same-budget independent baseline, FINAL (3 seeds/side)**
(`reports/baseline_comparison_gap_check_20260912.md`). The honest result required
several corrections. An initial no-augmentation comparison showed the supernet's
`large` level beating `fast_scnn` (the one comparable-budget baseline, 1.136M vs.
~1.047M params) by 2.6-6.4 mIoU on all 5 splits; that was superseded once `fast_scnn`
gained 9.0-11.5 mIoU points from augmentation alone — more than the supernet's entire
original margin. Augmented-vs-augmented comparisons with 1 and then 2 seeds each
narrowed the margin further and flipped which side "won" more splits, confirming a
single-seed-pair result should not be trusted. With the full 3 seeds per side
`RESEARCH_PLAN.md` §9 rule 8 requires:

| dataset | fast_scnn-aug (3-seed avg) | supernet-large-aug (3-seed avg) | gap |
|---|---:|---:|---:|
| Cityscapes | 0.5228 | 0.5338 | +0.0110 |
| ACDC/fog | 0.5629 | 0.5650 | +0.0021 |
| ACDC/night | 0.3822 | 0.3757 | −0.0065 |
| ACDC/rain | 0.5050 | 0.5027 | −0.0023 |
| ACDC/snow | 0.5048 | 0.5055 | +0.0007 |

Supernet wins 3/5 splits, loses 2/5, every margin **≤1.1 mIoU points** (a near-exact
tie on ACDC/snow). This is the stable, headline-ready result: **true near-parity**,
comfortably inside RQ2's hypothesized ±1.0-1.5 mIoU band and clear of the §11 no-go
trigger (>2 mIoU loss) either direction. **The claim for the paper is that the shared
supernet matches independent same-budget training within ~1 mIoU point across clean
and all four adverse conditions, at a fraction of the training/maintenance cost of
training one model per device/budget — RQ2's actual hypothesis, confirmed, not
exceeded.** Earlier "wins 4/5" (1 seed) and "wins 2/5" (2 seeds) framings should not
be cited; only the 3-seed table above is the citable number.

**Side finding, not yet understood**: augmentation does not help every architecture.
`segformer_b0` scored *lower* with augmentation than without (Cityscapes 0.5665 ->
0.5513, -0.0152), the opposite of `fast_scnn`'s large gain.
`bisenetv2`/`ddrnet23_slim`/`mobilenetv3_deeplabv3` have not been re-trained with
augmentation, so whether this is a transformer-vs-CNN effect or specific to
SegFormer's architecture/hyperparameters is unknown.

**[TODO: pending augmentation-regression investigation]** Before writing any general
"augmentation improves robustness" claim, the `segformer_b0` regression needs a
per-class breakdown or an augmentation-strength ablation, per
`reports/baseline_comparison_gap_check_20260912.md`'s side-finding section.

**Latency (measured, not FLOPs)** — the complete cross-backend table
(`reports/edge/E1_hailo_benchmark_protocol_20260910.md`,
`reports/edge/E3_tensorrt_benchmark_all_levels_20260911.md`), end-to-end mean over
3 runs per level:

| level | E1 Hailo-8 (ms) | E3 TensorRT GPU FP16 (ms) |
|---|---:|---:|
| tiny | 3.714 | 0.912 |
| small | 6.597 | 1.781 |
| medium | 20.29 | 4.546 |
| large | 41.14 | 9.389 |

**Pareto subnet selection** (`reports/pareto_search_v1_20260912.md`): joining this
latency table with the mIoU table above, all four levels are Pareto-optimal on both
devices (no crossovers). Under an identical 10 ms budget, the best level differs by
device — `small` on E1 (Hailo) vs. `large` on E3 (TensorRT GPU) — because E3 is
roughly 4-4.5x faster per level. This is concrete first evidence that hardware-
aware selection is not interchangeable with a one-size-fits-all subnet choice
(RQ1's premise), though not yet a full test of RQ1's quantitative hypothesis.

**[TODO: pending FLOPs-aware baseline, RQ1]** No FLOPs-aware selection baseline
exists to test RQ1's "≥15-20% cost reduction vs. FLOPs-aware search" hypothesis;
the Pareto result shows device-dependence, not a quantified win margin
(`reports/pareto_search_v1_20260912.md`).

3/3 seeds per side are now augmented (`RESEARCH_PLAN.md` §9 rule 8 satisfied) — the
go/no-go table above is headline-ready; no further seeds needed for this specific
comparison.

**[TODO: pending remaining baselines]** `segformer_b0` is trained/evaluated
(README.md Phase 8) but not yet in this comparison table; `pidnet_s` is skipped
(DDRNet-23-slim satisfies "PIDNet-S or DDRNet-23-slim"); `hard`/`ucpnet` remain
unimplemented (no public code/checkpoint found).

**[TODO: pending TensorRT E2/E5 data]** The Pareto table covers only E1 and E3; E2
and E5 latency data does not exist yet.

## Ablations and failure analysis

`RESEARCH_PLAN.md` §10 specifies a fixed ablation set: shared supernet vs.
independent training; FLOPs vs. measured-latency objective; latency-only vs.
latency+energy; unconstrained vs. compiler-safe operator space; PTQ vs. QAT;
with/without distillation; static-large/small vs. router; entropy vs. calibrated
router; per-frame vs. windowed routing; and behavior across 5/10/15/30 W power
modes. One finding anticipates the "with/without distillation" ablation: the
supernet's original (superseded) advantage over `fast_scnn` was plausibly
attributable to in-place distillation and the boundary-aware loss acting as
implicit regularization — an unverified hypothesis, not yet an ablation result.

**[TODO: pending Phase 9, "not started" per README.md]** None of the ten ablations
above has been run. The DLA-demotion decision
(`reports/edge/E3_compiler_smoke_test_20260908.md`) is the closest thing to a
completed "unconstrained vs. compiler-safe operator space" finding, but it is a
compiler-compatibility result, not the accuracy/cost ablation §10 specifies.

## Limitations

Results to date cover only two of the plan's four intended device classes as
*primary* backends — Xavier DLA was demoted to a secondary, encoder-only ablation
after a confirmed 16-subgraph-per-DLA-core hardware limit, not a software defect
later toolchains might lift, so any "DLA support" claim must stay scoped to the
encoder. E5 (Orin Nano Super, JetPack 6, no DLA) stands in for the plan's original
"legacy Nano" slot; whether that substitution is acceptable, or a true legacy (EOL
JetPack 4, Maxwell) board is still needed, is an open decision
(`RESEARCH_PLAN.md` §14, `docs/PARTNER_BRIEFING.md` §3). Calibration metrics
implemented so far (ECE, NLL, Brier, AURC) are prerequisites for a router that does
not exist yet — no "reliable" or "safety" language is warranted until it is built
and, per §11, actually improves external-shift behavior ("safety" language is
disallowed by the plan regardless). All latency numbers are backend- and
compiler-version-locked (TensorRT 8.5.2.2, HailoRT 4.23.0/DFC 3.34.0), not
guaranteed under a different toolchain release. Energy/J-frame is entirely
unclaimed pending an external calibrated power meter — a deliberate, disclosed
deferral, not an oversight.

**[TODO: pending device-lineup decision]** The Nano-vs-E5 framing above is
unresolved and must be settled before the device lineup is described definitively.

## Conclusion

**[TODO: pending nearly everything above]** A conclusion cannot honestly be written
yet: it needs go/no-go outcomes across all four RQs, and today only RQ2 has a
headline-ready (3-seed), one-baseline signal (near-parity with independent training,
not a win — see Experiments; still only 1 of the 7 baseline slots is a valid
same-budget test), RQ1 only a qualitative illustration, and RQ3/RQ4 have no router or
systematic compiler-space comparison respectively. Draft this section last, after
Phases 6 (QAT training run + real INT8-vs-FP32 numbers), 7 (router) and 9 (ablations)
land.
