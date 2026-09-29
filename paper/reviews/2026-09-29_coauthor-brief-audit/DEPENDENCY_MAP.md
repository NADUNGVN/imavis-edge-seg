# PACE-Seg evidence dependency map

This map separates source measurements, derived artifacts, manuscript claims, and
figures. README is navigation only.

## RQ1: measured hardware cost

`outputs/flops_by_level.json` + `outputs/benchmark_lookup_table.csv`
-> `reports/flops_baseline_v1_20260917.md`
-> `reports/rq1_budget_sweep_v1_20260920.md`
-> `paper/figures/scripts/plot_flops_latency.py`
-> Figure 3 and RQ1 prose.

## RQ2: shared-training practical near-parity

Three supernet evaluation JSON files + three Fast-SCNN evaluation JSON files
-> `reports/baseline_comparison_gap_check_20260912.md`
-> `paper/figures/scripts/plot_rq2_near_parity.py`
-> Figure 4 and RQ2 prose.

The statistical unit is the training run. The observed mean differences are
descriptive and remain within the prespecified +/-1.5 mIoU practical band; the
analysis is not a formal equivalence test.

## RQ3: deployment-matched routing

Run A/B/C checkpoint + licensed validation data
-> `scripts/evaluate_router.py`
-> per-run evaluation JSON and per-image dump with all-pixel entropy
-> `scripts/replay_router_with_overhead.py`
   + `reports/router_overhead_E1_20260928.json`
   + `reports/router_overhead_E3_20260922.json`
-> six run/backend replay JSON files
-> `scripts/summarize_router_replays.py`
-> canonical deployment-matched summary JSON
-> router results table, Figure 6, abstract, discussion, and conclusion.

The complete chain is orchestrated by
`scripts/run_router_deployment_matched_pipeline.py`. Each new artifact records the
run label, source checkpoint and hash, code commit, risk-feature definition, split
protocol, cost source, backend, and budget grid.

The corrected evaluator selects representative A--D operating points from fit-half
metrics only. Its low-cost pooled-calibrator baseline uses the same fit caches and is
replayed with the same E1/E3 costs; it tests whether the primary condition-specific
calibration is necessary.

### Historical artifacts invalidated for the final RQ3 claim

The files below remain immutable historical diagnostics, but their fit-time entropy
used a ground-truth-valid mask unavailable at deployment:

- `reports/router_per_image_dump_seed0.json`
- `reports/router_per_image_dump_seed2.json`
- `reports/router_per_image_dump_seed3.json`
- `reports/router_overhead_replay_E1_20260928.json`
- `reports/router_overhead_replay_E3_20260922.json`
- `reports/router_overhead_replay_E{1,3}_seed{2,3}_20260929.json`

Because logits were not saved, corrected entropy scores cannot be reconstructed from
these dumps. Full inference is mandatory.

## Secondary QAT study

FP32 and QAT evaluation artifacts
-> `reports/qat_rescue_2x2_screen_infra_20260920.md`
-> secondary QAT table and limitations prose.

The result is fake-quantized PyTorch accuracy, not compiled INT8 deployment accuracy.
