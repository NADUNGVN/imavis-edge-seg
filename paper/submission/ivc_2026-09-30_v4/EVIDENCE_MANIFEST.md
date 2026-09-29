# Evidence and figure manifest

## Immutable evidence references

- Deployment-matched method commit: `ababda12a9bfb6a5f92a7d79aad3560d361f863f`
- Deployment-matched result commit: `b57fcb2`
- Manuscript snapshot tag: `ivc-deployment-matched-v4-20260930`
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

### Figure 3: FLOPs versus measured latency

- Generator: `paper/figures/scripts/plot_flops_latency.py`
- Inputs: `outputs/flops_by_level.json`, `outputs/benchmark_lookup_table.csv`
- Canonical reports: `reports/flops_baseline_v1_20260917.md`,
  `reports/rq1_budget_sweep_v1_20260920.md`
- Encoding: four actual capacities; measured mean end-to-end latency; large/tiny scaling.

### Figure 4: RQ2 near parity

- Generator: `paper/figures/scripts/plot_rq2_near_parity.py`
- Supernet inputs: `reports/eval_pace_seg_v1_aug_seed0_step100000.json`,
  `reports/eval_pace_seg_v1_seed2_step100000.json`,
  `reports/eval_pace_seg_v1_aug_seed3_step100000.json`
- Fast-SCNN inputs: `reports/eval_baseline_fast_scnn_aug_step100000.json`,
  `reports/eval_baseline_fast_scnn_seed1_aug_step100000.json`,
  `reports/eval_baseline_fast_scnn_seed2_aug_step100000.json`
- Canonical report: `reports/baseline_comparison_gap_check_20260912.md`
- Whiskers are observed cross-run envelopes, not confidence intervals or seed pairing.

### Figure 5: deployment-matched routing

- Generator: `paper/figures/scripts/plot_router_deployment_matched.py`
- Inputs: all six `run_{a,b,c}_{e1,e3}_replay.json` files in the canonical RQ3 directory.
- Derived plot metrics: `paper/figures/generated/fig5_router_deployment_matched_metrics.json`.
- Encoding: macro held-out mIoU versus observed complete warm-route cost; each point
  averages five splits within a run and then Runs A--C; whiskers are run-level ranges.
- The generator verifies the deployment-matched all-pixel risk-feature metadata.

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
