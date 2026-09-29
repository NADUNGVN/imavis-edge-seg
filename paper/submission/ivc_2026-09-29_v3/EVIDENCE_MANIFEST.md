# Evidence and figure manifest

## Immutable evidence references

- Frozen evidence snapshot: `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`
- Canonical evidence baseline: `1c4d42a7ea525931154ebb2f9015d898ef7771a5`

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

## Conditional router evidence

Canonical routing artifacts remain listed in the previous package. Their final arithmetic
is 120 operating cells, 69 fair cells, D versus A 51/16/2, macro +0.0241 mIoU, D 0/120
and A 51/120 violating cells. These values are retained only as a conditional diagnostic
because fitting and deployment use different entropy pixel sets. They are not visualized
as a final quality--latency figure in v3.

## Claim status

- RQ1 measured-cost claim: supported within the four-candidate, four-device setting.
- RQ2 near-parity claim: supported as a descriptive three-run comparison.
- RQ3 routing mechanism: retained; quantitative advantage remains conditional.
- QAT: secondary fake-quant evidence only.
