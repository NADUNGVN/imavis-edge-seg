# Reviewer #2 audit: scientific-story revision

## A. Recommendation

Major Revision

## B. Confidence

5/5

## C. Scores

| Criterion | Score |
|---|---:|
| Novelty | 6/10 |
| Technical soundness | 6/10 |
| Experimental rigor | 7/10 |
| Hardware validity | 8/10 |
| Reproducibility | 6/10 |
| Clarity | 8/10 |
| Significance | 7/10 |

## D. Fatal flaws / reject-level issues

No writing-fixable reject-level issue remains. One experimental issue prevents the central RQ3 numerical result from becoming a submission headline: the calibration feature differs between fitting and deployment because the fit-half score uses a ground-truth ignore mask.

## E. Major concerns

1. RQ3 is a primary methodological contribution, but its measured advantage is conditional on a mismatched calibration feature. The manuscript must not imply that the reported +0.0241 mIoU is a deployment-matched estimate.
2. The 69 fair cells depend on policy A's feasibility and must remain paired with full-grid violation counts.
3. E1 and E3 share prediction caches. Cross-backend results validate deployment-cost sensitivity, not independent accuracy replication.
4. The three router checkpoints are separate seed-labelled runs, but immutable hashes and fully controlled identical-configuration provenance are incomplete.

## F. Minor concerns

1. Complete route costs are measured on two backends, while candidate-only latency is measured on four devices.
2. Per-condition calibrators assume that the deployment condition is known.
3. Compiler safety has no trained unconstrained-operator control.
4. QAT evidence is fake quantization and should remain secondary.

## G. Claim-by-claim audit

| Manuscript claim | Classification | Supporting artifact or action |
|---|---|---|
| Four compiler-safe static capacities | VERIFIED FACT | export/compiler reports and implementation; bounded as an engineering property |
| 110.2x FLOPs range versus 10.3--19.5x latency range | VERIFIED FACT | `reports/flops_baseline_v1_20260917.md` and latency lookup table |
| Proportional FLOPs proxy mis-selects measured choices | VERIFIED FACT | `reports/rq1_budget_sweep_v1_20260920.md` |
| Elastic large is within 0.0110 mIoU of Fast-SCNN | VERIFIED FACT | `reports/baseline_comparison_gap_check_20260912.md` |
| Candidate-specific routing improves deployment quality | UNVERIFIED CLAIM if stated without qualification | Retain only as a conditional diagnostic until risk features match |
| 67/69 fair-cell wins or ties and +0.0241 mIoU | VERIFIED FACT for the frozen replay | Canonical E1/E3 replay JSON files; not a deployment-matched estimate |
| D has 0/120 violating cells | VERIFIED FACT with structural qualification | Canonical replay; hard feasibility constraint partly determines the result |
| EMA-percentile QAT bounds degradation to 1.5 points | VERIFIED FACT for fake quantization | `reports/qat_rescue_2x2_screen_infra_20260920.md` |
| Compiled INT8 accuracy | MISSING EVIDENCE | Removed from manuscript claims |
| Energy improvement | MISSING EVIDENCE | Removed from manuscript claims |

## H. Numerical-consistency audit

The manuscript uses 120 total cells, 69 fair cells, 51 wins, 16 ties, 2 losses, 67/69 wins-or-ties, macro +0.0241 mIoU, D 0/120 violating cells, and A 51/120 violating cells. No stale 80-cell or pre-overhead aggregate is presented as the overhead-aware result. RQ1, RQ2, route-cost, and QAT values match the evidence manifest.

## I. Leakage and fit/held-out audit

Calibrators, risk targets, and representative operating points use the fit half. Held-out outcomes are read after operating-point selection. The oracle is not a deployable input. The ground-truth ignore mask used in fit-time score computation does not leak held-out labels, but it creates a train--deployment feature mismatch. This distinction is now explicit in the abstract, Method, Experimental Protocol, Results, Limitations, and figure caption.

## J. Hardware/latency validity audit

Warm route cost includes probe inference, risk computation, policy work, backend switching or activation, and additional candidate inference with tiny-output reuse. E1 and E3 have different runtime semantics and cold costs are not pooled. Accuracy is replayed from cached trained-model predictions rather than measured from the compiled timing engines; the manuscript states this boundary.

## K. Statistical-dependence audit

Operating cells reuse images, predictions, thresholds, and candidate sets. The paper uses descriptive counts without treating cells as independent samples. It does not attach p-values, confidence intervals, or reliability probabilities to 67/69 or 0/120.

## L. Novelty and closest-work audit

The manuscript positions efficient segmentation, elastic subnetworks, dynamic inference, hardware-aware design, and uncertainty calibration as inherited directions. The claimed contribution is their integration into candidate-specific, measured-budget runtime selection with complete route-cost validation. The wording does not claim that elasticity, calibration, boundary supervision, or hardware-aware search originates in this work.

## M. Minimum required fixes before submission

1. Compute probe entropy with the same pixel definition during fitting and deployment.
2. Refit calibrators and risk grids using the locked fit halves.
3. Replay the held-out evaluation without changing split membership or selecting on held-out outcomes.
4. Re-audit every RQ3 number and promote it to the abstract only if the conclusion survives.

## N. Optional improvements

1. Add a qualitative panel selected by a prespecified rule from real licensed outputs.
2. Add checkpoint hashes and immutable training manifests for future runs.
3. Evaluate a pooled calibrator or condition selector.
4. Measure integrated compiled-model accuracy and energy if those claims are desired.
