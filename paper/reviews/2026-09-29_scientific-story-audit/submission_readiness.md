# Submission-readiness statement

## Unresolved blocker

The calibration feature is not identical at fitting and deployment. Fit-half entropy excludes ground-truth ignore pixels; deployment entropy cannot. A matched-feature refit and held-out replay are required before the RQ3 numerical result can become a central submission claim.

Author affiliation, corresponding-author email, funding, conflict-of-interest, and CRediT statements must also be supplied before journal submission. They are not inferred in the manuscript source.

## Acceptable limitations

- Per-condition calibration assumes that the condition is known.
- Operating cells are correlated descriptive analysis points.
- E1 and E3 share segmentation predictions.
- Complete route cost is measured on two structurally different backends.
- Compiled timing engines and cached accuracy predictions are not a single integrated execution.
- Training provenance lacks checkpoint hashes and complete immutable manifests.
- QAT is fake quantization only.
- Energy is not measured.

## Claims removed or weakened

- The router arithmetic was removed from the abstract and conclusion headline.
- Cross-backend accuracy replication was replaced by deployment-cost robustness.
- Zero budget violations are described as partly structural, not reliability.
- Compiler safety is an engineering property, not an isolated accuracy contribution.
- QAT is a secondary bounded observation, not deployed INT8 evidence.

## Exact canonical routing diagnostic

The frozen replay contains 120 correlated operating cells and 69 fair cells. D versus A on fair cells is 51 wins, 16 ties, and 2 losses, or 67/69 wins-or-ties, with macro +0.0241 mIoU. Across the full grid, D has 0/120 violating cells and A has 51/120. These values are retained as conditional diagnostic evidence only.

## Immutable snapshots

- Frozen evidence: commit `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`, tag `gpt-review-v1-20260929`.
- Scientific-story manuscript: tag `ivc-scientific-story-v2-20260929`.

## Readiness decision

The source package is ready for Overleaf compilation and another external review round. It is not yet recommended for journal submission with RQ3 as a central empirical claim until the deployment-matched calibration rerun is audited.
