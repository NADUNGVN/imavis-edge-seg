# Evidence manifest for the packaged manuscript

## Immutable references

- Manuscript release tag: `ivc-scientific-story-v2-20260929`
- Frozen evidence snapshot: `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`
- Canonical evidence baseline: `1c4d42a7ea525931154ebb2f9015d898ef7771a5`

## Canonical routing artifacts

- `reports/router_overhead_replay_E3_20260922.json`
- `reports/router_overhead_replay_E1_20260928.json`
- `reports/router_overhead_replay_E3_seed2_20260929.json`
- `reports/router_overhead_replay_E1_seed2_20260929.json`
- `reports/router_overhead_replay_E3_seed3_20260929.json`
- `reports/router_overhead_replay_E1_seed3_20260929.json`
- `reports/router_overhead_v1_20260922.md`
- `reports/audit_gpu_risk_kernel_production_E3.json`

## Canonical supporting reports

- `reports/baseline_comparison_gap_check_20260912.md`
- `reports/pareto_search_v1_20260912.md`
- `reports/flops_baseline_v1_20260917.md`
- `reports/rq1_budget_sweep_v1_20260920.md`
- `reports/qat_v1_20260913.md`
- `reports/calibrated_qat_v1_20260917.md`
- `reports/qat_rescue_2x2_screen_infra_20260920.md`
- `reports/router_progressive_ablation_v1_20260921.md`

## Routing diagnostic arithmetic

- Total operating cells: 120.
- Fair operating cells: 69.
- D versus A on fair cells: 51 wins, 16 ties, 2 losses.
- Win-or-tie count: 67/69.
- Macro D-minus-A: +0.0241 mIoU, or +2.41 points.
- Budget-violating cells: D 0/120; A 51/120.

These values are canonical for the frozen replay artifacts, but they are not treated as a frozen deployment headline in manuscript v2. Fit-half entropy excludes ground-truth ignore pixels, whereas held-out and runtime entropy include all pixels. This is not held-out-label leakage, but it is a fit--deployment feature mismatch. RQ3 remains conditional until the calibration feature is computed identically at fitting and deployment.

## Manuscript claim status

- RQ1 measured-cost claim: ready within the tested four-candidate, four-device setting.
- RQ2 shared-training near-parity claim: ready as a descriptive three-run comparison.
- RQ3 candidate-specific routing claim: method contribution retained; quantitative advantage reported only as a conditional diagnostic.
- QAT claim: secondary fake-quant evidence only; no compiled INT8 accuracy claim.

## Repository documentation fixes closed after review

1. Tag `gpt-review-v1-20260929` now identifies authoring snapshot `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`.
2. The stale seed0 E3 filename in `reports/router_overhead_v1_20260922.md` has been corrected to canonical `reports/router_overhead_replay_E3_20260922.json`.

Both changes are documentation and reproducibility fixes; no experimental evidence was modified.
