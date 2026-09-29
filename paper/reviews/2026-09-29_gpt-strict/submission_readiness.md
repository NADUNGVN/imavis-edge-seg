# PACE-Seg Submission Readiness

## Current status

The scientific manuscript has completed one authoring pass, an independent Reviewer #2 audit, a mandatory revision pass, and a second independent audit. No unresolved reject-level issue remains for the narrowed claims, and no new experiment is required before submission.

The second audit recommends **Minor Revision** because two repository-documentation defects remain. The paper should not be labelled fully submission-ready until they are corrected and the manuscript is transferred to the target journal template.

## Unresolved blockers

1. The repository documentation advertises tag `gpt-review-v1-20260929`, but that tag does not resolve. Create it at the intended frozen snapshot or replace all tag references with commit `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`.
2. `reports/router_overhead_v1_20260922.md` refers to nonexistent seed0 E3 file `reports/router_overhead_replay_E3_20260928.json`. Replace it with the committed canonical file `reports/router_overhead_replay_E3_20260922.json`.

These are reproducibility/documentation fixes, not requests for new data.

## Unresolved but acceptable limitations

- Calibration is fitted separately for Cityscapes and each ACDC condition; the evaluation assumes that the deployment domain or condition is known or configured.
- Fit-half entropy uses a ground-truth ignore mask while deployment entropy does not. This is a fit/deployment feature mismatch, not held-out leakage.
- The final result is offline per-image replay of cached trained-model predictions using directly measured complete warm-route costs. It is not a trained compiled-engine accuracy run.
- E1 and E3 share segmentation predictions. Their comparison establishes cross-backend cost/deployment behavior, not independent accuracy replication.
- Operating cells reuse images, thresholds, predictions, and candidate sets. The 120 cells are correlated and are not a statistical sample size.
- Fair-cell inclusion is conditional on A having no held-out budget violation. The 69-cell quality result must stay paired with all-120-cell violation counts.
- D's zero violating-cell count is partly structural because D enforces a hard candidate-cost constraint.
- Three separately trained seed-labelled checkpoints are available, but complete checkpoint hashes, config hashes, initialization parents, and proof of identical configurations are unavailable.
- QAT evidence uses PyTorch fake quantization, not deployed compiled INT8 execution.
- No external power meter was used, so no energy claim is admissible.

## Claims intentionally removed or weakened

- “Reliable edge vision” and universal robustness language were removed.
- “Device-conditioned decisions” was replaced by **hardware-cost-conditioned routing**; different devices need not select different candidates.
- “Three independent seeds” was weakened to **three separately trained seed-labelled checkpoints**.
- Cross-backend replication was limited to deployment-cost behavior because E1 and E3 share predictions.
- QAT was limited to fake-quant degradation; compiled INT8 accuracy and speedup are not claimed.
- RQ1 was limited to failure of a simple through-origin proportional FLOPs proxy in this setting, not FLOPs-aware NAS in general.
- The router was limited to per-domain/per-condition calibration; unknown-condition routing is not claimed.
- EMA-percentile outlier suppression is described as a plausible mechanism, not a measured causal explanation.

## Canonical headline result

Using directly measured complete warm-route costs, candidate-specific, hardware-cost-conditioned policy D outperformed or matched rank policy A in **67 of 69 conditionally fair operating cells**: **51 wins, 16 ties, and 2 losses**. The descriptive macro difference was **+0.0241 mIoU**, equivalent to **+2.41 mIoU points**. Across all **120 correlated operating cells**, D produced **0/120** budget-violating cells and A produced **51/120**. The zero count is partly due to D's hard constraint. The two backends share segmentation predictions.

## Intended immutable submission snapshot

- Authoring snapshot: `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`
- Canonical evidence baseline inside that snapshot: `1c4d42a7ea525931154ebb2f9015d898ef7771a5`
- Expected freeze label: `gpt-review-v1-20260929` (currently unresolved and therefore not yet an acceptable immutable citation)

## Canonical router artifacts

- `reports/router_overhead_replay_E3_20260922.json`
- `reports/router_overhead_replay_E1_20260928.json`
- `reports/router_overhead_replay_E3_seed2_20260929.json`
- `reports/router_overhead_replay_E1_seed2_20260929.json`
- `reports/router_overhead_replay_E3_seed3_20260929.json`
- `reports/router_overhead_replay_E1_seed3_20260929.json`
- `reports/router_overhead_v1_20260922.md`
- `reports/audit_gpu_risk_kernel_production_E3.json`

## Final handoff criterion

After the two repository-documentation blockers are fixed, the next work is editorial rather than experimental: transfer the Markdown manuscript into the Image and Vision Computing/Elsevier template, generate figures from canonical artifacts, format references, and perform page-level proofreading without changing the frozen claims or numbers.
