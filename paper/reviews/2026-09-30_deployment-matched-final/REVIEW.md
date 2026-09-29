# Independent Reviewer #2 audit — deployment-matched revision v4

## A. Recommendation

**Minor Revision.** No reject-level scientific issue remains within the manuscript's
stated scope. The remaining mandatory items are submission metadata and a final
repository-snapshot pin, not new experiments.

## B. Confidence

**5/5.** The review used the canonical raw JSON, manifest hashes, implementation,
derived figure metrics, LaTeX source, and rendered PDF.

## C. Scores (1--10)

| Criterion | Score |
|---|---:|
| Novelty | 7 |
| Technical soundness | 8 |
| Experimental rigor | 8 |
| Hardware validity | 8 |
| Reproducibility | 9 |
| Clarity | 8 |
| Significance | 7 |

## D. Fatal flaws / reject-level issues

None found after the deployment-matched rerun. The former fit/deployment feature
mismatch is repaired in code and in the final artifacts rather than explained away.

## E. Major concerns

1. **VERIFIED FACT — offline hardware replay.** Accuracy predictions and hardware
   timings are not produced by one integrated trained compiled engine. The manuscript
   states this in the latency protocol and limitations. This bounds deployment
   validity but does not invalidate a replay claim.
2. **VERIFIED FACT — correlated cells.** The 120 cells reuse images, predictions,
   thresholds, and candidate sets. The manuscript treats the result descriptively,
   reports Runs A--C as the training axis, and does not use cell-level significance.
3. **VERIFIED FACT — incomplete initialization provenance.** Checkpoint SHA-256 and
   embedded configuration hashes are now recorded, but initialization-parent records
   are unavailable. The manuscript no longer calls the runs a perfectly controlled
   identical-configuration seed study.

These are explicit limitations, not hidden defects. None requires a new experiment
for the present claims.

## F. Minor concerns

- A qualitative input/ground-truth/prediction panel remains unavailable because the
  repository does not redistribute licensed RGB images and masks. It would improve
  accessibility but is not required to support the quantitative claims.
- Cold-start semantics differ between TensorRT and Hailo and are excluded.
- Pooled calibration uses fit-half examples from all evaluated conditions; it does
  not establish performance on an unseen condition.
- The title and abstract are appropriately bounded, but submission metadata still
  needs author affiliation, corresponding-author details, declarations, and the
  journal's final checklist.

## G. Claim-by-claim audit

| Claim location | Classification | Evidence | Audit outcome |
|---|---|---|---|
| Title: hardware-cost-conditioned elastic segmentation | VERIFIED FACT | Four measured candidate costs per backend; policy code | Retain; does not imply device decisions always differ |
| Abstract: FLOPs scaling mismatch | VERIFIED FACT | FLOPs JSON, latency LUT, RQ1 reports | Retain with proportional-proxy boundary |
| Abstract/RQ2: within 1.1 mIoU points | VERIFIED FACT | Six evaluation JSON files and baseline report | Retain as near parity, not superiority |
| Abstract/RQ3: 70/74, +0.0284, 0 versus 46 | VERIFIED FACT | Canonical six replay JSON files and recomputed summary | Retain with fair-cell and correlation boundaries |
| Pooled calibration removes condition selection | VERIFIED FACT | Pooled replay records | Retain for evaluated conditions |
| Pooled result implies condition-specific fitting is unnecessary here | REASONABLE INFERENCE | 58/14/2, +0.0296; configured-versus-pooled comparison | Wording is correctly local, not universal |
| Complete route costs are backend-dependent | VERIFIED FACT | E1/E3 measurement JSON and harnesses | Retain; warm paths only |
| EMA-percentile stabilizes fake-quant degradation | VERIFIED FACT | QAT rescue report | Retain as secondary fake-quant evidence |
| Outlier sensitivity explains QAT rescue | REASONABLE INFERENCE | Observer comparison only | Manuscript correctly labels mechanism unmeasured |
| Universal reliability or compiled INT8 deployment | UNVERIFIED CLAIM | No supporting evidence | Not present in v4 |

## H. Numerical-consistency audit

Recomputation from the six final replay files gives:

- total cells: 120;
- fair cells: 74;
- configured D W/T/L: 56/14/4;
- configured D win-or-tie: 70/74 = 94.6%;
- configured D macro delta: 0.0283548767 mIoU;
- configured D violating: 0/120;
- A violating: 46/120;
- pooled D W/T/L: 58/14/2;
- pooled D macro delta: 0.0295858635 mIoU;
- pooled D violating: 0/120.

The abstract, Table 4, Results, Conclusion, evidence manifest, and Figure 5 derived
metrics agree after rounding. Historical 69/120 and +0.0241 values do not appear as
current evidence in the v4 manuscript.

## I. Leakage and fit/held-out audit

- `compute_deployment_risk_score` averages entropy over all output pixels and is used
  during prediction collection for both fit and held-out records.
- Candidate error targets use only non-ignore ground-truth labels on the fit half.
- Risk and entropy grids use fit-half values only.
- Representative risk targets are selected from fit-half quality and latency.
- Held-out labels enter only final confusion-matrix aggregation, violation reporting,
  diagnostics, and the explicitly non-deployable oracle.
- Pooled calibrators combine fit halves only; they do not consume held-out outcomes.

Conclusion: no held-out leakage was found in the final protocol. The historical
feature mismatch is neither present nor used by the headline.

## J. Hardware/latency validity audit

The E3 path includes tiny inference, device-resident entropy, policy work, dispatch,
and additional candidate inference. The E1 path includes mandatory activation and
deactivation, tiny inference, host entropy, policy work, and additional inference.
Tiny output is reused. Replays substitute complete route costs for the old
candidate-only LUT and reselect operating points on fit-half statistics. E1 and E3
cold scenarios are correctly excluded as non-equivalent. The production CUDA audit
uses real engine outputs and the actual PyTorch risk function; the reported maximum
error is a tested-sample correctness result, not an agreement probability.

## K. Statistical-dependence audit

No p-value, confidence interval, or significance wording treats operating cells as
independent. Figure 5 whiskers are explicitly the range across three run-level macro
values. The two backends reuse segmentation predictions and are described as two
deployment-cost validations rather than accuracy replications. Fair-cell conditioning
is defined in Methods and paired with full-grid violation counts.

## L. Novelty and closest-work audit

The manuscript does not claim elastic networks, dynamic routing, hardware-aware NAS,
or calibration individually. Relative to Once-for-All and SlimSeg, the contribution is
runtime image-wise selection under directly measured complete route costs. Relative to
Learning Dynamic Routing and anytime dense prediction, the distinguishing mechanism is
candidate-specific expected-error mapping from one shared probe plus an explicit hard
hardware budget. Relative to AutoSegEdge and hardware-aware segmentation NAS, the work
addresses runtime selection among already exposed candidates rather than design-time
architecture search. The novelty is therefore integrative, methodological, and
deployment-evaluative; the wording matches that scope.

## M. Minimum required fixes before submission

1. Add final author affiliation, corresponding-author email, funding, conflict-of-
   interest, data/code availability, and CRediT statements required by Elsevier.
2. Create and push the declared immutable tag `ivc-deployment-matched-v4-20260930`
   on the revision commit.
3. Upload the v4 Overleaf ZIP and confirm the journal template compiles identically.

No new experiment is required for the claims currently made.

## N. Optional improvements

- Add the deterministic qualitative panel if licensed source images can be included.
- Evaluate calibration on an adverse condition absent from the pooled fit set.
- Run an integrated trained compiled-engine accuracy path.
- Add cold-start and temporal-window evaluation.
- Add external power-meter measurements before making energy claims.

## Submission-readiness statement

No unresolved reject-level blocker remains. Acceptable limitations are offline replay,
shared predictions across backends, correlated operating cells, validation-half
evaluation, incomplete initialization provenance, and no compiled INT8 or energy
claim. Claims intentionally removed or weakened include universal reliability,
independent cross-backend accuracy replication, condition-specific calibration as the
source of the routing gain, and deployed INT8 accuracy. The canonical headline is
configured D versus A: 120 total cells, 74 fair, 56/14/4 W/T/L, +0.0283549 mIoU,
D 0/120 and A 46/120 violating; pooled D is 58/14/2 and +0.0295859. The evidence
snapshot is result commit `b57fcb2`; the manuscript snapshot is the declared tag
`ivc-deployment-matched-v4-20260930`.
