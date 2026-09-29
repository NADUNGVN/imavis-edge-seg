# Independent Reviewer #2 Report after Mandatory Revision

Manuscript reviewed: `PACE_SEG_MANUSCRIPT_GPT_FINAL.md`

Evidence snapshot audited: `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`

Canonical evidence baseline: `1c4d42a7ea525931154ebb2f9015d898ef7771a5`

## A. Recommendation

**Minor Revision**

The revision resolves the prior claim-level major concern: it now states throughout that calibration is fitted separately for Cityscapes and each ACDC condition, and that deployment therefore assumes a known or configured domain. No reject-level methodological problem remains for the paper's narrowed claims. The remaining mandatory items are repository-freeze and filename corrections that do not require new experiments.

## B. Confidence

**5/5**

## C. Scores

| Dimension | Score (1--10) | Assessment |
|---|---:|---|
| Novelty | 6 | A defensible integrative and deployment-oriented contribution, not a claim that the constituent techniques are new. |
| Technical soundness | 7 | The canonical replay preserves fit/held-out separation; limitations of per-condition calibration and feature mismatch are explicit. |
| Experimental rigor | 7 | Negative results, three checkpoint runs, two measured backends, and complete warm-route accounting are strengths; cells are correlated and provenance remains incomplete. |
| Hardware validity | 7 | E1 and E3 expose distinct real overhead mechanisms. The accuracy path remains offline replay rather than trained compiled-engine execution. |
| Reproducibility | 6 | Canonical artifacts and code resolve, but the documented freeze tag is absent and one repository report retains a stale filename. |
| Clarity | 8 | The problem--gap--method--evidence chain is coherent and key denominators and boundaries are stated where used. |
| Significance | 6 | The result is useful for bounded edge-segmentation routing; generality to unknown conditions, larger candidate sets, or other devices is open. |

## D. Fatal flaws / reject-level issues

None remain for the claims actually made in the revised manuscript.

The paper would again become indefensible if it were changed to claim condition-agnostic routing, independent E1/E3 accuracy replication, trained compiled-engine accuracy, deployed INT8 accuracy, energy savings, formal reliability, or 120 independent trials.

## E. Major concerns

No unresolved major concern requires a new experiment.

The main scientific boundary is now correctly represented: each split uses its own fit-half calibrators. This is a per-domain/per-condition protocol, not a router that discovers an unknown adverse condition. The abstract, Introduction, Related Work positioning, Method, Results, Discussion, Limitations, and Conclusion all preserve that boundary.

The fit-time entropy feature excludes ground-truth ignore pixels while held-out/runtime entropy does not. The manuscript correctly classifies this as a train/deployment feature mismatch rather than held-out leakage.

The hardware result is correctly described as overhead-aware offline replay with directly measured complete warm-route costs. The timing engines and cached prediction source are no longer conflated.

## F. Minor concerns

1. Create the repository tag advertised by the review documentation, or replace every reference to it with the actual immutable commit.
2. Correct `router_overhead_replay_E3_20260928.json` to `router_overhead_replay_E3_20260922.json` in the canonical overhead report.
3. If available before submission, record checkpoint and configuration hashes for the three router runs. Their absence is disclosed and does not invalidate the narrowed result.
4. In the journal-formatted version, convert raw URLs to the venue's bibliography style and add access/version metadata for software artifacts.

## G. Claim-by-claim audit

| Claim location and substance | Classification | Evidence | Reviewer conclusion |
|---|---|---|---|
| Title: hardware-cost-conditioned elastic segmentation | VERIFIED FACT | `router/policy.py`; E1/E3 overhead artifacts | Accurate; it does not say every device must choose differently. |
| Abstract: four elastic levels | VERIFIED FACT | `src/imavis_edge_seg/config.py` | Supported. |
| Abstract/Method: candidate-specific risk from one tiny probe | VERIFIED FACT | `risk_probe.py`, `calibrator.py`, `policy.py` | Supported, with per-condition calibrators now explicit. |
| Abstract/Results: 67/69 fair cells, 51/16/2 | VERIFIED FACT | FINAL LOCK report and six replay JSON files | Supported as descriptive correlated-cell counts. |
| Abstract/Results: macro +0.0241 mIoU | VERIFIED FACT | Same | Supported; correctly identified as 2.41 points and not an independent-sample estimate. |
| Abstract/Results: D 0/120 versus A 51/120 violating cells | VERIFIED FACT | Same | Supported; structural role of D's hard constraint is stated. |
| Abstract/Method: condition identity known/configured | VERIFIED FACT about protocol | Per-split fitting in `evaluate_router.py` | Correct and essential qualification. |
| Introduction: measured route costs differ from abstract complexity | VERIFIED FACT | FLOPs/Pareto/RQ1 reports | Narrowed to a proportional through-origin FLOPs proxy. |
| Contributions: integrated method and deployment audit | REASONABLE INFERENCE | Method plus related-work comparison | Defensible integrative novelty; no unsupported priority claim. |
| Method: fit/held-out separation in final replay | VERIFIED FACT | `replay_router_with_overhead.py` | Canonical replay selects operating points using fit-half statistics. |
| Method: older progressive D path is noncanonical | VERIFIED FACT | `evaluate_router.py` versus final replay | Correctly disclosed. |
| Results: large level within 1.1 points of Fast-SCNN | VERIFIED FACT | baseline comparison report | Supports near-parity only. |
| Results: 110.2x FLOPs versus 10.3--19.5x latency | VERIFIED FACT | FLOPs and device reports | Applies only to the four candidates and devices. |
| Results: complete warm E1/E3 route costs | VERIFIED FACT | overhead scripts and JSON | Includes probe, risk, policy, activation/dispatch, and extra inference; tiny reused. |
| Results: CUDA kernel numerical audit | VERIFIED FACT | production audit script and JSON | Forty outputs, max error 7.15e-7; observed decision agreement is not a population probability. |
| Results: EMA-percentile QAT stays within 1.5 points | VERIFIED FACT | QAT rescue report | Fake quant only; no compiled INT8 implication. |
| Discussion: candidate-specific calibration helps in tested protocol | REASONABLE INFERENCE | A/D comparison | Appropriately bounded to known domains and evaluated budgets. |
| Conclusion: method can improve runtime selection | REASONABLE INFERENCE | Canonical replay | Wording is no broader than evidence. |
| Universal reliability or robustness | Not claimed | N/A | Correctly excluded. |
| Energy saving | Not claimed | N/A | Correctly excluded. |

## H. Numerical-consistency audit

All mandatory headline checks pass:

- 3 checkpoint runs x 2 backends x 5 splits x 4 budgets = 120 operating cells.
- Fair cells: 35 E3 + 34 E1 = 69.
- W/T/L: 51 + 16 + 2 = 69.
- Wins or ties: 67/69 = 97.1% after rounding.
- Macro difference: +0.0241 mIoU = +2.41 points.
- D violating: 0/60 + 0/60 = 0/120.
- A violating: 25/60 + 26/60 = 51/120.
- E3 warm medians: 2.07, 5.23, 11.97, 26.70 ms.
- E1 warm medians: 34.95, 46.13, 63.36, 92.68 ms.
- EMA-percentile worst losses in the large level: 0.97, 0.86, and 1.38 points across the three listed runs.

The 80-cell and 52-fair-cell figures remain only in the explicitly labelled seed0, candidate-only, pre-overhead ablation. No stale +0.0256 value appears. The final evidence filenames in the manuscript resolve to the committed six replay files.

## I. Leakage and fit/held-out audit

Calibrators use even-index fit-half labels to construct candidate-error targets. Risk targets are derived from fit-half error quantiles. The final replay evaluates A and D separately on fit and held-out caches and chooses the representative operating point from fit-half mean latency and mIoU. Odd-index held-out labels are not used to fit calibrators, define thresholds, or choose deployable operating points.

The older progressive evaluator's held-out-based D selection is not used in the final headline. The oracle uses held-out ground truth only as a nondeployable upper bound. The corrected replay uses one canonical risk grid per checkpoint run across E1 and E3; seed0's grid is not reused for seed2 or seed3.

Per-condition fitting is not held-out leakage, but it supplies condition identity at the protocol level. The manuscript now states this assumption. Fit-time masked entropy versus runtime unmasked entropy is a feature mismatch, not leakage.

## J. Hardware/latency validity audit

E3 measurements include synchronized tiny TensorRT inference, device-resident entropy, policy computation, context dispatch, and extra selected inference. E1 measurements include Hailo network-group activation/deactivation, tiny inference, host entropy, policy computation, and extra selected inference. Both reuse the tiny output when tiny is selected.

Warm route classes are directly timed rather than assembled by summing unrelated candidate medians. Backend-specific cold semantics are not pooled. The Hailo artifacts are fresh post-rescale HEFs, and the production CUDA audit exercises real TensorRT tensors and the production PyTorch risk function.

The accuracy path uses cached trained-model predictions, while the compiled timing graphs characterize route cost. The paper discloses this distinction and does not claim bit-exact trained compiled accuracy.

## K. Statistical-dependence audit

The manuscript contains no p-value, confidence interval, or significance statement that treats operating cells as independent. It explains that thresholds and budgets reuse images and predictions, and that E1/E3 share prediction caches. The 67/69 count is not described as a success probability. The three seed-labelled checkpoints are the main training axis, but the text avoids claiming proven identical-configuration independence.

## L. Novelty and closest-work audit

Once-for-All, SlimSeg, and MESS already establish elastic or multi-exit candidate families. Learning Dynamic Routing and Anytime Dense Prediction establish adaptive dense inference. MnasNet, FasterSeg, AutoSegEdge, and related work establish hardware-aware architecture design. Calibration literature establishes confidence/error analysis for segmentation.

PACE-Seg's defensible difference is narrower: per-domain candidate-specific error maps from one probe, hard feasibility under directly measured complete route costs, and explicit two-backend overhead characterization. This is an integrative, deployment-oriented contribution. The manuscript does not use “first,” “unique,” or claim ownership of its inherited components.

## M. Minimum required fixes before submission

No additional experiment is required for the current claims.

Two documentation fixes remain mandatory:

1. make the advertised immutable repository tag resolve, or replace tag references with commit `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`;
2. correct the stale E3 seed0 replay filename in `reports/router_overhead_v1_20260922.md`.

The manuscript-level mandatory revisions from the first review are complete: condition-specific calibration, known-domain deployment scope, ignore-mask mismatch, offline replay boundary, cautious three-run terminology, checkpoint provenance table, exact AdamW optimizer, official SlimSeg DOI, and paired fair/all-cell denominators.

## N. Optional improvements that are not required for the present claims

- Evaluate one pooled, deployment-matched calibrator on mixed or unseen conditions.
- Record full checkpoint/config hashes and resume lineage.
- Execute trained compiled engines and compare their outputs with PyTorch.
- Report UIoU with ACDC invalid-region labels.
- Evaluate temporal routing, switching hysteresis, and cold frontiers.
- Add externally metered energy per frame.
- Compare with a learned latency predictor.

## Final reviewer judgment

The manuscript now describes the same bounded study that the canonical evidence supports. Its remaining weaknesses are explicit limitations rather than hidden contradictions. After the two repository-documentation corrections and ordinary journal formatting, the work is suitable to enter submission rather than another experiment cycle.
