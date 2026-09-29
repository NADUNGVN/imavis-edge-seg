# Evidence manifest for the packaged manuscript

## Immutable references

- Manuscript authoring snapshot: `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`
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

## Canonical headline arithmetic

- Total operating cells: 120.
- Fair operating cells: 69.
- D versus A on fair cells: 51 wins, 16 ties, 2 losses.
- Win-or-tie count: 67/69.
- Macro D-minus-A: +0.0241 mIoU, or +2.41 points.
- Budget-violating cells: D 0/120; A 51/120.

## Repository documentation fixes still required

1. The advertised tag `gpt-review-v1-20260929` does not currently resolve. Create it at the intended snapshot or replace the tag references with the exact authoring commit.
2. In `reports/router_overhead_v1_20260922.md`, replace nonexistent `reports/router_overhead_replay_E3_20260928.json` with canonical `reports/router_overhead_replay_E3_20260922.json`.

These are documentation and reproducibility fixes. They do not require new experiments.
