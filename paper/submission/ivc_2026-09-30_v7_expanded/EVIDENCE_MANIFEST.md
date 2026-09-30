# Evidence and figure manifest

## Immutable evidence references

- Deployment-matched method commit: `ababda12a9bfb6a5f92a7d79aad3560d361f863f`
- Deployment-matched result commit: `b57fcb2`
- Parent manuscript snapshot tag: `ivc-qualitative-v6-20260930`
- Canonical RQ3 directory: `reports/router_deployment_matched_20260929/`

## Figure provenance

### Figure 1: complete PACE-Seg path

- Editable source: `paper/figures/archify/fig1_pace_seg_system/candidate.json`
- Journal vectors: `paper/figures/generated/fig1_pace_seg_system.{svg,pdf}`
- Validation record: `paper/figures/archify/VALIDATION.md`
- Code evidence: supernet, static extraction/export, entropy probe, candidate calibrators,
  and budgeted policy referenced inside the Archify source.

### Figure 2: elasticity and static deployment

- Editable source: `paper/figures/archify/fig2_elastic_deployment/candidate.json`
- Journal vectors: `paper/figures/generated/fig2_elastic_deployment.{svg,pdf}`
- Validation record: `paper/figures/archify/VALIDATION.md`
- Configuration evidence: `src/imavis_edge_seg/config.py` and static extraction/export code.

### Figure 3: dataset and condition overview

- Generator: `paper/figures/scripts/render_qualitative_examples.py`.
- Selection and render audit: `paper/figures/qualitative/selection_manifest.json` and
  `paper/figures/qualitative/generated/render_manifest.json`.
- Encoding: one input and ground-truth pair per split, selected nearest the held-out
  median large-candidate pixel error.

### Figure 4: FLOPs versus measured latency

- Generator: `paper/figures/scripts/plot_flops_latency.py`
- Inputs: `outputs/flops_by_level.json`, `outputs/benchmark_lookup_table.csv`
- Canonical reports: `reports/flops_baseline_v1_20260917.md`,
  `reports/rq1_budget_sweep_v1_20260920.md`
- Encoding: four actual capacities; measured mean end-to-end latency; large/tiny scaling.

### Figure 5: RQ2 near parity

- Generator: `paper/figures/scripts/plot_rq2_near_parity.py`
- Supernet inputs: `reports/eval_pace_seg_v1_aug_seed0_step100000.json`,
  `reports/eval_pace_seg_v1_seed2_step100000.json`,
  `reports/eval_pace_seg_v1_aug_seed3_step100000.json`
- Fast-SCNN inputs: `reports/eval_baseline_fast_scnn_aug_step100000.json`,
  `reports/eval_baseline_fast_scnn_seed1_aug_step100000.json`,
  `reports/eval_baseline_fast_scnn_seed2_aug_step100000.json`
- Canonical report: `reports/baseline_comparison_gap_check_20260912.md`
- Whiskers are observed cross-run envelopes, not confidence intervals or seed pairing.

### Figure 6: real-image routing decision explainer

- Generator: `paper/figures/scripts/plot_routing_decision_example.py`.
- Audited image source: `paper/figures/qualitative/generated/fig7_qualitative_grid.png`,
  whose SHA-256 is verified against `render_manifest.json` before cropping.
- Risk source: candidate-specific Run A rain values in the render manifest.
- Cost sources: `reports/router_overhead_E3_20260922.json` and
  `reports/router_overhead_E1_20260928.json`.
- Derived audit: `paper/figures/generated/fig7_routing_decision_example_audit.json`.
- Boundary: the shared 60 ms budget is an explanatory derived scenario, not a
  canonical headline operating cell or an independent evaluation trial.

### Figure 7: deployment-matched routing

- Generator: `paper/figures/scripts/plot_router_deployment_matched.py`
- Inputs: all six `run_{a,b,c}_{e1,e3}_replay.json` files in the canonical RQ3 directory.
- Derived plot metrics: `paper/figures/generated/fig5_router_deployment_matched_metrics.json`.
- Encoding: macro held-out mIoU versus observed complete warm-route cost; each point
  averages five splits within a run and then Runs A--C; whiskers are run-level ranges.
- The generator verifies the deployment-matched all-pixel risk-feature metadata.

### Figure 8: qualitative capacity comparison

- Generator: `paper/figures/scripts/render_qualitative_examples.py`.
- Input IDs and output hashes: `paper/figures/qualitative/generated/render_manifest.json`.
- Checkpoint: Run A SHA-256
  `a24441568a4cd14e5aabfbcad833d4797ee61a7921bf631994ba1c3f26274ad2`.
- The renderer regenerated every selected prediction and required exact agreement
  with all four canonical per-image confusion matrices before writing the panels.
- At the displayed E3 budget, policy D selects medium for all five examples; the
  figure is not used to claim diverse routing decisions.

### Figure S1: predetermined failure case

- Selection rule: maximum Run A large-candidate pixel error across the four ACDC
  held-out halves, fixed before visual inspection.
- Image ID: `GOPR0351_frame_000825`; large-candidate pixel error 0.4435869.
- This is a deliberately worst case and is not treated as representative.

### Figure S2: complete-route cost expansion

- Generator: `paper/figures/scripts/plot_route_cost_expansion.py`.
- Candidate-only p95 source: `outputs/benchmark_lookup_table.csv`.
- Complete warm-route median sources: `reports/router_overhead_E3_20260922.json`
  and `reports/router_overhead_E1_20260928.json`.
- Derived metrics: `paper/figures/generated/figS2_route_cost_expansion_metrics.json`.
- Boundary: the plot compares two measured totals and does not claim a measured
  component-level decomposition.

## Visual-data addendum

### Tables 1--3

- Candidate-family source: `paper/tables/candidate_family.csv`; FLOPs trace to
  `outputs/flops_by_level.json`, candidate-only p95 latency to
  `outputs/benchmark_lookup_table.csv`, and architecture dimensions to
  `src/imavis_edge_seg/config.py` plus the static subnet implementation.
- Dataset-count source: `paper/tables/dataset_splits.csv`, verified from the four
  versioned CSV manifests under `data/manifests/`.
- Policy source: `paper/tables/policy_summary.csv`, traced to
  `src/imavis_edge_seg/router/policy.py` and the deployment-matched replay protocol.

### Table S1: condition-level router comparison

- Generator: `paper/figures/scripts/summarize_router_by_condition.py`.
- Machine-readable source: `paper/tables/router_by_condition.csv`.
- LaTeX table: `paper/tables/router_by_condition_table.tex`.
- Inputs: all six canonical E1/E3 deployment-matched replay JSON files.
- Fair-cell and tie definitions exactly match `scripts/summarize_router_replays.py`.

### Real-data figure audit

- Frozen selection: `paper/figures/qualitative/selection_manifest.json`.
- Selection generator: `paper/figures/scripts/select_qualitative_examples.py`.
- Audited server renderer: `paper/figures/scripts/render_qualitative_examples.py`.
- Primary rule: one held-out image nearest the median large-candidate pixel error per
  split, plus one separately labelled highest-error ACDC case.
- Selected IDs: `frankfurt_000000_003920`, `GP010476_frame_000031`,
  `GOPR0356_frame_000324`, `GOPR0402_frame_000748`,
  `GOPR0122_frame_000352`; hardest case `GOPR0351_frame_000825`.
- Final raster outputs and their SHA-256 values are recorded in
  `paper/figures/qualitative/generated/render_manifest.json`. The repository contains
  only composite publication figures; raw datasets and checkpoints remain absent.

## Canonical router evidence

The final deployment-matched arithmetic is 120 operating cells and 74 fair cells.
Configured D versus A records 56/14/4 wins/ties/losses, a macro +0.0283549 mIoU,
D 0/120 violating cells, and A 46/120. Pooled D uses the same fair subset and records
58/14/2 with +0.0295859 mIoU and 0/120 violating cells.

The primary machine-readable sources are:

- `reports/router_deployment_matched_20260929/manifest.json`;
- `reports/router_deployment_matched_20260929/summary.json`;
- three deployment-matched evaluation JSON files and per-image dumps;
- six E1/E3 complete-route replay JSON files.

The manifest records the code commit, checkpoint SHA-256 values, command chain, and
SHA-256 values for every output. Historical masked-fit artifacts remain auditable but
are not used by the manuscript headline.

The evaluator defines the fit-time and deployment-time feature identically as mean
softmax entropy over all output pixels. Candidate error targets use only non-ignore
ground-truth labels on the fit half. Held-out outcomes are read after risk-target
selection and are not used to fit calibrators or choose operating points.

### Manuscript-facing run mapping

- Run A: `pace_seg_v1_aug_seed0`, step 100,000.
- Run B: `pace_seg_v1_seed2`, step 100,000.
- Run C: `pace_seg_v1_aug_seed3`, step 100,000.

Full checkpoint SHA-256 and embedded configuration hashes are stored in the canonical
manifest and evaluation metadata. Initialization-parent provenance remains unavailable.

### Deployment-matched rerun chain

- Evaluator: `scripts/evaluate_router.py`.
- E1/E3 replay: `scripts/replay_router_with_overhead.py`.
- Descriptive aggregation: `scripts/summarize_router_replays.py`.
- End-to-end driver: `scripts/run_router_deployment_matched_pipeline.py`.
- Low-cost routing control: pooled-across-splits candidate calibrators, evaluated
  under the same hard-budget policy and complete route costs.
- External rerun inputs: the three checkpoints and licensed Cityscapes/ACDC data.
- Output rule: use a new directory; never overwrite the historical replay artifacts.

## Claim status

- RQ1 measured-cost claim: supported within the four-candidate, four-device setting.
- RQ2 near-parity claim: supported as a descriptive three-run comparison.
- RQ3 routing result: supported by deployment-matched held-out replay with measured
  complete warm-route costs; correlated-cell and offline-replay boundaries apply.
- QAT: secondary fake-quant evidence only.
