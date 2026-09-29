# Scientific-story audit

## Scope

This round restructures the PACE-Seg manuscript for *Image and Vision Computing*, Special Issue: Complex Environment Vision. It treats the manuscript as a frozen scientific study rather than an experimental chronology.

## Input and output

- Repository input snapshot: `e81a3c1cbd0d04c608d783161fb50a525f766858`.
- Prior manuscript: `paper/submission/ivc_2026-09-29/`.
- Revised manuscript: `paper/submission/ivc_2026-09-29_v2/`.
- Experimental reports and raw result files are unchanged.

## Structural decisions

The paper is organized around three questions:

1. whether directly measured hardware cost is necessary;
2. whether shared elastic training preserves segmentation quality; and
3. whether candidate-specific calibrated risk improves budgeted routing.

Only the elastic candidate family, measured-cost selection, and candidate-specific budgeted routing are primary contributions. QAT remains a secondary deployment study. Backend implementation details are retained only where they define the measurement protocol or a claim boundary.

## Outcome

RQ1 and RQ2 are presented as bounded findings. RQ3 remains a primary method contribution, but its numerical comparison is explicitly diagnostic because fit-time entropy excludes ground-truth ignore pixels while deployment-time entropy includes all pixels. The mismatch is not held-out-label leakage, but it prevents a deployment-matched router headline.

No experimental artifact or numerical result was changed in this round.
