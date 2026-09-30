# Manuscript table sources

These CSV files are the compact machine-readable sources for the new onboarding
tables in the v5 manuscript.

- `candidate_family.csv`: resolution, width, and depth are defined in
  `src/imavis_edge_seg/config.py`; FLOPs come from `outputs/flops_by_level.json`;
  trainable parameters are counted from the static subnet architecture including
  batch-normalization affine parameters; E1/E3 values are candidate-only p95 latency
  from `outputs/benchmark_lookup_table.csv`.
- `dataset_splits.csv`: counts are verified against `data/manifests/*.csv`; the
  alternating-index fit/held-out split is implemented by `scripts/evaluate_router.py`.
- `policy_summary.csv`: properties follow `src/imavis_edge_seg/router/policy.py` and
  the deployment-matched protocol in `scripts/replay_router_with_overhead.py`.

Candidate-only latency in the first table must not be confused with complete warm
route cost, which additionally includes the probe, entropy, policy, activation or
dispatch, and any second inference.
