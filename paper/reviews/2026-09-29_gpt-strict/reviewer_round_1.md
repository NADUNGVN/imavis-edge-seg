# Independent Reviewer #2 Report on `PACE_SEG_MANUSCRIPT_GPT_V1.md`

## A. Recommendation

**Major Revision**

The frozen evidence supports a bounded systems paper, but the first manuscript version still understates one deployment assumption: each dataset split or adverse condition receives its own fitted calibrators. The manuscript can be made defensible without new experiments by narrowing the router claim to domain-specific calibration, documenting the evaluation pipeline precisely, and correcting repository reproducibility defects. A condition-agnostic router, compiled-model accuracy replication, or formal reliability guarantee would require new evidence and must not be implied.

## B. Confidence

**5/5**

## C. Scores

| Dimension | Score (1--10) | Rationale |
|---|---:|---|
| Novelty | 6 | Candidate-specific mappings plus measured hard-budget routing are a useful integration, but the ingredients and adjacent dynamic-inference ideas are established. |
| Technical soundness | 6 | Final replay separation is sound, but per-condition calibration and the fit/runtime entropy mismatch constrain interpretation. |
| Experimental rigor | 7 | The project exposes negative results, uses fit/held-out separation, multiple training runs, and measured overhead. Cells remain correlated and run provenance is incomplete. |
| Hardware validity | 7 | Route costs include the relevant warm-path components on two distinct backends. Accuracy is replayed offline rather than produced by the timed compiled engines. |
| Reproducibility | 6 | Code and canonical reports are extensive, but the advertised tag is absent, one FINAL LOCK filename is stale, and checkpoint/config hashes are incomplete. |
| Clarity | 7 | The manuscript is substantially clearer than the repository draft, but condition-specific calibration must be explicit throughout. |
| Significance | 6 | The deployment lesson is useful for a four-candidate segmentation system; generality beyond that setting is not demonstrated. |

## D. Fatal flaws / reject-level issues

No unavoidable reject-level flaw remains if the claims are narrowed as specified below.

The following would become reject-level if retained or introduced:

- **UNVERIFIED CLAIM:** a single condition-agnostic calibrator routes unknown Cityscapes/ACDC conditions. `scripts/evaluate_router.py::_evaluate_split` fits `fit_per_level_calibrators` separately for every split.
- **UNVERIFIED CLAIM:** E1 and E3 independently replicate accuracy. They replay the same per-image predictions with different cost tables.
- **UNVERIFIED CLAIM:** the paper validates trained compiled-engine segmentation accuracy. Timing engines have the correct graph shape, while accuracy comes from cached PyTorch checkpoint predictions.
- **UNVERIFIED CLAIM:** three controlled seeds use an identical immutable training configuration. The checkpoint names include two augmentation-labelled runs and one earlier unlabelled run, and full provenance hashes are not canonicalized.
- **UNVERIFIED CLAIM:** fake-quant QAT demonstrates deployed INT8 accuracy.

## E. Major concerns

### E1. Per-condition calibrators are a central deployment assumption

**VERIFIED FACT.** `_collect_split_data` is called per split, and `_evaluate_split` fits a fresh set of per-level calibrators from that split's fit half. The final replay serializes and reuses those split-specific calibrators. ACDC/fog, night, rain, and snow are therefore evaluated with calibrators fitted using labels from the same named condition.

This is not held-out leakage: odd-index held-out labels do not fit the policy. It is, however, domain-conditioned calibration. A deployment that does not know the current condition cannot select the demonstrated calibrator without an external condition detector or a pooled calibrator that has not been evaluated. The title can remain hardware-cost-conditioned, but the abstract, method, results, discussion, and conclusion must say that calibration is fitted separately per evaluation domain/condition.

### E2. Fit-time and runtime probe features differ

**VERIFIED FACT.** `src/imavis_edge_seg/router/risk_probe.py::compute_risk_score` optionally excludes `IGNORE_INDEX`. `scripts/evaluate_router.py::_collect_split_data` passes the ground-truth mask for fit-half images and passes `None` for held-out images. Thus the calibrator is trained on masked entropy and queried with unmasked entropy.

This is not leakage because no held-out label enters the policy. It is a train/deployment feature mismatch. The held-out performance is still empirical evidence that the mismatched pipeline works in this sample, but calibration semantics are weaker than claimed by phrases such as deployment-matched expected error. The manuscript correctly discloses the mismatch in Sections 3.2 and 7; it should also avoid describing the calibration protocol as fully matched in the abstract.

### E3. Hardware validation is component-integrated latency plus offline replay

**VERIFIED FACT.** `scripts/measure_router_overhead.py` and `scripts/measure_router_overhead_hailo.py` time complete route classes, including probe inference, entropy, decision logic, activation/dispatch, and additional candidate inference. `scripts/replay_router_with_overhead.py` then applies those route costs to cached confusion matrices and scores.

The timed graphs do not produce the trained predictions used for quality. This is a valid hardware-cost replay, but not an end-to-end on-device application run with trained engine outputs. The first manuscript version discloses this in Sections 4.3 and 7. It should use “overhead-aware offline replay using directly measured route costs,” not an unqualified “deployed routing accuracy.”

### E4. Training-run provenance is insufficient for “independent three-seed replication”

**VERIFIED FACT.** The FINAL LOCK lists `pace_seg_v1_seed0`, `pace_seg_v1_seed2`, and `pace_seg_v1_aug_seed3`, while other reports and dumps use `pace_seg_v1_aug_seed0`. The repository establishes distinct seed-labelled run IDs and checkpoints but does not provide a final table with checkpoint SHA-256, config hash, initialization parent, augmentation flag, and resume lineage for every run.

The manuscript prudently says “separately trained seed-labelled checkpoints.” This wording must be retained. A paper table should list the exact run ID, seed, checkpoint path, augmentation flag, step, and available config hash. Missing hashes should be stated as unavailable rather than inferred.

### E5. The fair-cell result is conditional and zero violations are partly structural

**VERIFIED FACT.** The 69-cell denominator includes only cells where A has no per-image violation. This removes baseline comparisons in which A gains quality by exceeding budget and is arguably favorable to A, but it remains a post-hoc conditional subset rather than a randomized design. D's policy filters candidates by `latency_ms <= budget`, making zero violations largely expected for the evaluated breakpoint budgets.

The paper must always pair 67/69 with D 0/120 and A 51/120, explain both denominators, and avoid statistical language such as success probability. The first manuscript version does this adequately.

### E6. Repository freeze references are not immutable as advertised

**VERIFIED FACT.** `docs/GPT_REVIEW_GUIDE.md`, `docs/CLAIMS_EVIDENCE_MATRIX.md`, and `docs/GPT_STRICT_REVIEW_PROMPT.md` refer to tag `gpt-review-v1-20260929`, but the repository tag page contains no tag. Commit `ac65b562ed1cc33b7a30a63b2451924b156f8bdd` is the actual readable snapshot and identifies evidence baseline `1c4d42a7ea525931154ebb2f9015d898ef7771a5`.

**VERIFIED FACT.** The FINAL LOCK block in `reports/router_overhead_v1_20260922.md` names `reports/router_overhead_replay_E3_20260928.json`, but the committed seed0 E3 artifact is `reports/router_overhead_replay_E3_20260922.json`. The absent filename produces a 404.

Before submission, create the documented immutable tag or replace tag references with the real commit, and correct the stale E3 filename in documentation. This is a reproducibility fix, not a new experiment.

### E7. Novelty is integrative and deployment-oriented

**REASONABLE INFERENCE.** Once-for-All, SlimSeg, and MESS already provide elastic or multi-exit segmentation candidates. Learning Dynamic Routing and Anytime Dense Prediction already implement input-adaptive dense inference. MnasNet, FasterSeg, AutoSegEdge, and RealtimeSeg already incorporate measured or predicted hardware latency. Segmentation-calibration papers already map confidence to error behavior.

The defensible novelty is the combination of one shared probe, candidate-specific expected-error mappings, a hard measured route-cost constraint, and explicit two-backend route-overhead accounting. “First,” “unique,” and claims that individual components originate here should be removed.

### E8. The FLOPs baseline is deliberately simple

**VERIFIED FACT.** RQ1 evaluates a through-origin proportional FLOPs-to-latency proxy, not a learned latency predictor or modern hardware-aware NAS. The strong failure rates are valid for that proxy. The manuscript correctly limits the conclusion, but the abstract and conclusion must continue to say “simple proportional FLOPs proxy.”

## F. Minor concerns

- Replace the SlimSeg arXiv reference with its ACM record and DOI `10.1145/3503161.3548191`.
- Use “mIoU points” whenever subtracting values on the 0--1 scale to prevent ambiguity.
- The phrase “Adam-compatible weight decay” is vague unless the optimizer is verified from the trainer. State the exact optimizer or omit its name.
- The architecture table should be tied to `src/imavis_edge_seg/config.py`; compiler-safe portability should not be called an accuracy improvement.
- The maximum-observer QAT failure should not be presented as proof of outlier sensitivity. The manuscript correctly labels that mechanism unmeasured.
- ACDC UIoU is not reported despite available invalid masks. This is optional because the paper does not claim uncertainty-aware segmentation performance.
- The Hailo warm route and TensorRT warm route have different switching semantics. They are valid backend-specific costs but should not be described as mechanically identical experiments.
- The two losses in the final 69 fair cells are correlated replays of ACDC/rain, not two independent failure events.

## G. Claim-by-claim audit

| Manuscript claim | Classification | Supporting artifact | Boundary or correction |
|---|---|---|---|
| Title: hardware-cost-conditioned elastic segmentation | VERIFIED FACT | `router/policy.py`; overhead report FINAL LOCK | Add domain-specific calibration qualification in abstract/method, not necessarily title. |
| Four width-depth-resolution candidates | VERIFIED FACT | `src/imavis_edge_seg/config.py` | None. |
| Candidate-specific error estimates from one probe | VERIFIED FACT | `router/calibrator.py`; `router/policy.py` | Calibrators are separate by candidate and also fitted separately by dataset condition. |
| Hard measured-cost constraint | VERIFIED FACT | `_select_by_risk_and_latency_budget` | Zero violations are partly by construction. |
| Complete warm route costs measured on E1/E3 | VERIFIED FACT | `measure_router_overhead*.py`; overhead JSON/report | Accuracy is offline replay, not produced by the timed engines. |
| Large elastic level within 1.1 points of Fast-SCNN | VERIFIED FACT | `reports/baseline_comparison_gap_check_20260912.md` | Near-parity only; do not quantify maintenance savings. |
| FLOPs 110.2x versus latency 10.3--19.5x | VERIFIED FACT | FLOPs, Pareto, and RQ1 sweep reports | Applies to four candidates and simple proportional proxy. |
| 120 total and 69 fair cells | VERIFIED FACT | `reports/router_overhead_v1_20260922.md` FINAL LOCK | Correlated cells, conditional subset. |
| 51/16/2 and 67/69 | VERIFIED FACT | Same | Not a statistical success probability. |
| Macro +0.0241 mIoU | VERIFIED FACT | Same | Macro descriptive aggregate; specify 2.41 points. |
| D 0/120 and A 51/120 violating cells | VERIFIED FACT | Same | Operating-cell counts, not number of violating images; D hard-constrained by design. |
| Two structurally different backends | VERIFIED FACT | E1/E3 harnesses and reports | Shared predictions mean deployment robustness, not accuracy replication. |
| Three separate training runs | VERIFIED FACT | FINAL LOCK run IDs | Identical-config independence is not fully proven. |
| One router handles unknown conditions | MISSING EVIDENCE | Code fits per-split calibrators | Rewrite as per-domain/per-condition calibration. |
| GPU kernel matches production PyTorch | VERIFIED FACT | `audit_gpu_risk_kernel_production.py`; audit JSON | State 40 outputs and 0/10 observed decision mismatches; no population rate. |
| EMA-percentile keeps every tested level/run within 1.5 points | VERIFIED FACT | QAT rescue report | Fake quant only; mean improvement and outlier mechanism are unverified. |
| Compiled INT8 accuracy | MISSING EVIDENCE | No trained compiled-INT8 accuracy artifact | Exclude claim. |
| Energy saving | MISSING EVIDENCE | No external meter | Exclude claim. |
| Universal reliability | UNVERIFIED CLAIM | AURC weak on night; no guarantees | Exclude word or qualify narrowly. |
| Measured route costs improve runtime selection | REASONABLE INFERENCE | Final replay and A/D comparison | Supported in evaluated domains, budgets, candidates, and training runs only. |

## H. Numerical-consistency audit

The following headline arithmetic is internally consistent:

- Total cells: 3 training runs x 2 backends x 5 splits x 4 budgets = 120.
- Fair cells: E3 35 plus E1 34 = 69.
- Wins, ties, losses: 51 + 16 + 2 = 69.
- Win or tie: 67 / 69 = 0.971014, reported as 97.1%.
- Macro delta: 0.0241 mIoU = 2.41 points on a 0--100 scale.
- D violating cells: 0/60 + 0/60 = 0/120.
- A violating cells: 25/60 + 26/60 = 51/120.

No stale `+0.0256` or 80-cell result is used as the final claim in the new manuscript. The 80-cell progressive result remains explicitly labelled pre-overhead seed0 evidence. The exact E3 warm medians are 2.07, 5.23, 11.97, and 26.70 ms; E1 medians are 34.95, 46.13, 63.36, and 92.68 ms. QAT large-level EMA-percentile worst losses are 0.97, 0.86, and 1.38 points for runs 0, 3, and 2.

## I. Leakage and fit/held-out audit

### Calibrators

Fit-half labels determine per-candidate error targets and an ignore-pixel mask for the fit feature. This is allowed supervised calibration. Held-out labels are not used to fit calibrators.

### Risk targets

The five risk thresholds are quantiles of fit-half observed error. One canonical grid is used per training run across E1 and E3. The corrected seed2/seed3 replay does not reuse seed0's grid.

### Operating-point selection

`replay_router_with_overhead.py::replay_e2e_aware` computes fit and held-out decisions separately and passes only fit-half mean latency and mIoU to `select_budget_matched_operating_point`. This is clean.

The older `evaluate_router.py::_evaluate_split` constructs D's initial `d_grid` only on held-out predictions and selects among it using held-out mIoU. That path violates the claimed discipline for D. The FINAL LOCK overhead replay corrects the problem by recomputing D on both halves. The manuscript must state that only the replay is canonical for the headline.

### Oracle

The oracle uses held-out ground-truth error only to define an upper bound and is not fed into A or D. No deployable-policy leakage occurs.

### Condition identity

Each condition receives a calibrator fitted on the same condition's fit half. This is not held-out leakage but is an unreported condition oracle unless deployment is explicitly assumed to know the domain. This is the most important protocol qualification.

### Ignore-mask mismatch

The fit feature uses a ground-truth ignore mask and the held-out feature does not. This is a feature mismatch, not label leakage into evaluation.

## J. Hardware and latency validity audit

- **VERIFIED FACT:** E3 route timing includes synchronized tiny inference, GPU entropy, D policy computation, dispatch, and selected inference. Tiny output is reused.
- **VERIFIED FACT:** E1 includes tiny inference, host entropy, policy computation, mandatory activation/deactivation, and selected inference. Tiny output is reused.
- **VERIFIED FACT:** Warm costs are measured as one trace rather than by summing independent benchmark medians.
- **VERIFIED FACT:** E3's custom kernel reads the device-resident output; the production audit uses real TensorRT outputs and the real PyTorch risk function.
- **VERIFIED FACT:** Hailo uses newly compiled post-rescale HEFs and actual HailoRT activation semantics.
- **VERIFIED FACT:** E1 and E3 cold scenarios are not equivalent and are excluded from the final replay.
- **REASONABLE INFERENCE:** Architecture-correct untrained weights are adequate for pure latency characterization because tensor shapes and execution paths are fixed. This does not establish trained compiled accuracy.
- **MISSING EVIDENCE:** A fully integrated on-device run in which the trained engine produces the exact prediction subsequently used by the router.

## K. Statistical-dependence audit

No cell-level significance test is admissible. Budgets within a split reuse images; thresholds reuse fitted calibrators; E1 and E3 reuse predictions; and fair-cell inclusion depends on A's held-out route behavior. The three training runs are the only meaningful top-level replication axis, but they are not fully documented as identical-config seeds. Descriptive W/T/L counts and macro deltas are acceptable with these boundaries. Terms such as statistically significant, confidence interval over 120 cells, independent backend replication, reliability rate, and guaranteed robustness must not appear.

## L. Novelty and closest-work audit

| Closest work | What already exists | PACE-Seg distinction | Novelty type |
|---|---|---|---|
| Once-for-All | Shared elastic candidates and measured-latency specialization | Image-wise post-training routing among segmentation candidates with complete route costs | Integrative/deployment |
| SlimSeg | Slimmable segmentation, distillation, boundary supervision | Candidate-specific calibrated routing and hardware route accounting | Integrative |
| MESS | Train-once segmentation candidates, device-aware exit configuration | Separate compiled candidates and one-probe expected-error maps under a hard measured budget | Method integration |
| Learning Dynamic Routing for Semantic Segmentation | Image-dependent dense routes and budget regularization | Post-hoc candidate selection rather than learned internal paths | Methodological distinction |
| Anytime Dense Prediction | Confidence-adaptive dense computation | Image-level candidate selection with candidate-specific risk | Methodological distinction |
| AutoSegEdge / RealtimeSeg / FasterSeg | Hardware-aware design-time architecture search | Runtime image-conditioned selection among fixed candidates | Deployment distinction |
| Segmentation calibration studies | Pixel confidence calibration and domain-shift analysis | One shared image-level probe mapped to each candidate's expected error | Narrow mechanism |
| EQ-Net / AdaBits | Quantized elastic or shared-weight networks | Supporting 8-bit fake-quant observer study for elastic segmentation | Evaluative, not primary novelty |

The contribution wording in the manuscript is narrower than or equal to the evidence if condition-specific calibration is added.

## M. Minimum required fixes before submission

1. State in the abstract, Method 3.2/3.4, Results 5.5, Discussion, and Limitations that calibrators are fitted separately for each dataset split or adverse condition.
2. Make the deployment assumption explicit: condition identity is known/configured, or the result is a per-domain evaluation rather than an unknown-condition router.
3. Retain the fit/runtime ignore-mask mismatch as a limitation and avoid fully deployment-matched calibration wording.
4. Retain “offline per-image replay with measured complete route costs”; do not call it trained compiled-engine accuracy.
5. Replace any “three independent seeds” statement with “three separately trained seed-labelled checkpoints” unless provenance is completed.
6. Add a run-provenance table listing exact IDs and known configuration differences.
7. Correct the nonexistent freeze tag or replace it everywhere with commit `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`.
8. Correct the stale seed0 E3 replay filename to `reports/router_overhead_replay_E3_20260922.json`.
9. Replace the SlimSeg arXiv reference with the ACM DOI.
10. Keep the fair-cell and all-cell denominators together wherever the headline is repeated.

No new experiment is mandatory for the revised, domain-specific, replay-based claim.

## N. Optional improvements that must not block submission

- Refit a single pooled calibrator with the deployment-time unmasked entropy feature and evaluate unseen mixed conditions.
- Run trained compiled engines and compare their outputs with PyTorch predictions.
- Evaluate ACDC UIoU using the official invalid masks.
- Measure cold/reload frontiers under backend-specific scenarios.
- Add an external calibrated power meter and report energy per frame.
- Compare against a learned latency predictor rather than only a proportional FLOPs proxy.
- Evaluate temporal-window routing and engine-switch hysteresis.
- Add bootstrap intervals over images or paired checkpoint-level summaries that respect dependence, without treating operating cells as independent.

## Research-gap coherence check

After the mandatory condition-calibration correction, the paper's five logical components align:

- **Gap:** elastic candidates exist but image-wise candidate selection is unresolved. **Contribution:** candidate-specific error mappings from one shared probe.
- **Gap:** abstract cost does not determine device feasibility. **Contribution:** measured route-cost hard constraints.
- **Gap:** routing studies can omit actual probe and switching overhead. **Contribution:** synchronized route timing on TensorRT/CUDA and Hailo-8.
- **Gap:** deployment claims often blur quantization simulation, backend replication, and reliability. **Contribution:** explicit bounded evidence and limitations.

