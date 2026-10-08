# 05 — Statistical validity audit

| ID | Topic | Finding | Class | Severity |
|---|---|---|---|---|
| S-1 | Unit of analysis | 120 "cells" reuse the same held-out images, predictions and thresholds across budgets and devices; win/tie/loss counts are not independent trials. Manuscript states this (§4.4, Limitations). | — | ok |
| S-2 | Bootstrap hierarchy | Image-level bootstrap resamples held-out images per split, shared across runs/devices/budgets: captures image sampling only. Nested bootstrap (200 reps) adds fit-half resampling and refitting. **Seed (training-run) variance is not in any interval.** With 3 seeds, report per-run deltas (cheap, existing data). | SUGGESTED IMPROVEMENT | MAJOR |
| S-3 | Split correlation | Alternating-index split over ACDC sequences (e.g., GOPR0351 frames 39, 143, 825) and Cityscapes cities places neighbouring frames in both fit and held-out halves → optimistic calibration/operating-point transfer. LOCO partially addresses condition shift, not frame correlation. Stated in Limitations. A sequence-grouped split re-run of the replay would need no new training. | SUSPECTED ISSUE | MAJOR |
| S-4 | Equivalence language | D − T-hard CI [−0.11, 0.09] → "no detectable difference", not equivalence (rule 6). Fix "adds nothing", "matches". | VERIFIED (wording) | MINOR |
| S-5 | Fair-cell selection | D − A uses 62 fair cells chosen by A's own violations — conditions on the comparator; reported transparently together with A-hard over all 120 cells. | — | ok |
| S-6 | Static comparison denominator | Includes infeasible-route cells where D violates (04 R-1). | VERIFIED ERROR | MAJOR |
| S-7 | Risk–metric alignment | Calibration target is per-image pixel error; evaluation is dataset mIoU. Spearman(pixel error, per-image 1−mIoU) = 0.23–0.58. Pixel error is dominated by large classes (road, building, sky), so the router may ignore small-class failures. A class-balanced risk (per-image 1−mIoU or frequency-weighted error) and alternative entropy statistics (e.g., top-decile entropy, boundary entropy) can be evaluated on the existing per-image dumps only if the dumps hold per-image confusion matrices (they do) — cheap analysis. | SUGGESTED IMPROVEMENT | MINOR |
| S-8 | Multiple comparisons | Many comparisons (cost statistic × policy × LOCO) reported with CIs for a subset only; descriptive framing is appropriate; avoid "significant". | — | ok |
| S-9 | Break-even | Linear α-scaling counterfactual; no uncertainty on break-even values. A bootstrap over images of the crossing point is cheap. | SUGGESTED IMPROVEMENT | MINOR |
| S-10 | RQ2 | 3 seeds, mean ± s.d.; no test claimed. Fine. | — | ok |
