# Deployment-matched router evaluation (FINAL)

## Status

This report supersedes the masked-fit router headline for RQ3. The historical
artifacts remain available for audit, but manuscript claims and Figure 5 use only
the deployment-matched package in `reports/router_deployment_matched_20260929/`.

- Method commit: `ababda12a9bfb6a5f92a7d79aad3560d361f863f`
- Result commit: `b57fcb2`
- Canonical manifest: `router_deployment_matched_20260929/manifest.json`
- Canonical summary: `router_deployment_matched_20260929/summary.json`

The manifest records the complete command chain, checkpoint SHA-256 values, and
SHA-256 values for every output. Recomputing the summary from the six replay JSON
files reproduces all aggregate values below.

## Corrected protocol

The tiny probe feature is mean softmax entropy over all output pixels during both
calibrator fitting and deployment. Ground truth is used only to compute per-image
candidate-error targets over non-ignore labels on the fit half. Each validation
split is partitioned by alternating index (even: fit; odd: held out).

For every training run, one risk-target grid is derived from fit-half errors and
reused across E1 and E3. A representative operating point is selected using fit-half
quality and cost only. Held-out outcomes are read after selection. The oracle is not
used by a deployable policy.

The replay combines cached trained-model predictions with directly measured complete
warm-route costs. It includes probe inference, risk computation, policy decision,
backend switching or activation, and additional candidate inference when required.
Tiny output is reused when tiny is selected.

## Primary result: configured condition-specific D versus A

A fair cell is one in which baseline A has zero held-out per-image budget violations.
Win/tie/loss counts and mean quality differences use only fair cells; violation counts
use the full grid.

| Backend | Fair cells | D W/T/L | Win or tie | Mean D-A mIoU | D violating | A violating |
|---|---:|---:|---:|---:|---:|---:|
| E3 TensorRT/CUDA | 38/60 | 29/7/2 | 94.7% | +0.0287 | 0/60 | 22/60 |
| E1 Hailo-8 | 36/60 | 27/7/2 | 94.4% | +0.0279 | 0/60 | 24/60 |
| Overall | 74/120 | 56/14/4 | 94.6% | +0.0284 | 0/120 | 46/120 |

Thus configured D outperforms or matches A in 70 of 74 fair cells. The exact macro
gap is `0.028354876730473537` mIoU.

## Pooled-calibration control

The pooled control fits one candidate-specific calibrator set from the five fit
halves and uses no condition-specific calibrator choice at inference. On the same
74-cell fair subset, pooled D records 58 wins, 14 ties, and 2 losses, with a mean
pooled-D-minus-A gap of `0.029585863544868102` mIoU. It has 0/120 violating cells.

Configured D and pooled D are identical in 88 of 120 cells. Across the full grid,
configured D is better in 6 cells and pooled D is better in 26; the mean configured
D-minus-pooled-D difference is -0.0008314 mIoU. The effect is small, but it rules out
the claim that condition-specific calibrator selection is necessary for the observed
gain. Both variants remain candidate-specific across elasticity levels.

## Run-level breakdown

| Run | Backend | Fair | W/T/L | Mean D-A mIoU | A violating |
|---|---|---:|---:|---:|---:|
| A | E1 | 13/20 | 10/3/0 | +0.0297 | 7/20 |
| A | E3 | 13/20 | 10/3/0 | +0.0297 | 7/20 |
| B | E1 | 11/20 | 9/2/0 | +0.0246 | 9/20 |
| B | E3 | 11/20 | 9/2/0 | +0.0246 | 9/20 |
| C | E1 | 12/20 | 8/2/2 | +0.0291 | 8/20 |
| C | E3 | 14/20 | 10/2/2 | +0.0311 | 6/20 |

Runs A--C are separately trained checkpoint records. The canonical artifacts include
checkpoint and embedded configuration hashes, but initialization-parent provenance
is unavailable; they are not described as a perfectly controlled identical-config
seed experiment.

## Adverse-condition diagnostics

Across Runs A--C, the large level averages 0.5311 held-out mIoU on Cityscapes and
0.3559 on ACDC/night. Probe-signal AURC averages 0.1411 over the fifteen run-split
evaluations and is weakest on night (three-run mean 0.2416).

Configured D's four fair-cell losses are two underlying Run C operating points---fog
and rain at the medium budget---replayed on both backends. Pooled D's two losses are
one Run B snow operating point replayed on both backends. These duplicated backend
cells share predictions and are not independent failures.

## Claim boundaries

Admissible claim:

> Using deployment-matched all-pixel entropy and directly measured complete warm-route
> costs, candidate-specific hard-budget routing outperformed or matched rank escalation
> in 70 of 74 fair operating cells, with a macro +0.0284 mIoU gap. It produced no
> budget-violating operating cell across the 120-cell grid, compared with 46 for the
> baseline. A pooled candidate-calibration control retained the result.

Required boundaries:

- The 74 fair cells are conditionally defined by baseline A feasibility.
- The 120 cells reuse datasets, predictions, thresholds, and candidate sets; they are
  correlated descriptive evaluations, not 120 independent samples.
- E1 and E3 share segmentation predictions. They validate two measured deployment-cost
  paths, not independent accuracy replication.
- D's zero violating-cell count is partly structural because candidates above the
  budget are removed by design.
- Accuracy and timing are combined in offline replay. The timing engines have the
  correct architecture but are not the trained engines that generated predictions.
- The result covers warm-route TensorRT/CUDA and Hailo-8 measurements, not equivalent
  cold-start semantics, energy, or unseen adverse conditions.

## Figure and derived audit

- Generator: `paper/figures/scripts/plot_router_deployment_matched.py`
- Vector figure: `paper/figures/generated/fig5_router_deployment_matched.pdf`
- Derived metrics: `paper/figures/generated/fig5_router_deployment_matched_metrics.json`

The generator reads all six canonical replay files directly, verifies the risk-feature
metadata, and derives both configured and pooled comparison summaries without copying
headline values into the plotting source.
