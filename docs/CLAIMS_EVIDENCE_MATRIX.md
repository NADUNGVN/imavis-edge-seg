# Claims–Evidence Matrix — PACE-Seg

Review snapshot: `gpt-review-v1-20260929`  
Evidence baseline: `1c4d42a7ea525931154ebb2f9015d898ef7771a5`

| Claim | Canonical evidence | Replication scope | Headline result | Required boundary |
|---|---|---|---|---|
| RQ2: one elastic supernet matches independently trained same-budget models | [Baseline comparison](../reports/baseline_comparison_gap_check_20260912.md) | 3 supernet seeds vs 3 augmented Fast-SCNN seeds; Cityscapes + four ACDC conditions | Supernet wins 3/5 splits and loses 2/5; every margin is within about 1.1 mIoU points | Claim near-parity, not superiority; training/maintenance savings must be quantified or phrased cautiously |
| RQ1: FLOPs is an unsafe cross-platform latency proxy | [FLOPs baseline](../reports/flops_baseline_v1_20260917.md), [budget sweep](../reports/rq1_budget_sweep_v1_20260920.md), [Pareto report](../reports/pareto_search_v1_20260912.md) | 4 levels × E1/E2/E3/E5; 40-budget sweep | FLOPs large/tiny ratio 110× versus measured latency 10–20×; transferred proxies frequently mis-select and can violate budgets | This is measured-latency selection, not continuous hardware-aware architecture training; energy is out of scope |
| EMA-percentile calibration rescues shared-supernet fake-quant QAT | [QAT rescue screen](../reports/qat_rescue_2x2_screen_infra_20260920.md), with history in [QAT v1](../reports/qat_v1_20260913.md) and [max calibration](../reports/calibrated_qat_v1_20260917.md) | 3 paired supernet seeds; all 4 elasticity levels; Cityscapes + ACDC conditions | Every evaluated level/seed stays within the predeclared 1.5-point worst-condition degradation budget; pathological large-level seed improves from −2.54 to −1.38 points | Fake-quant evidence only; do not claim compiled-engine INT8 accuracy or universal mean-accuracy improvement; observer mechanism is consistent with outlier sensitivity, not directly proven |
| Candidate-specific hardware-cost-conditioned router D improves the risk–latency trade-off | [Progressive ablation](../reports/router_progressive_ablation_v1_20260921.md) and its canonical JSON artifacts | 3 independently trained supernets; E1/E2/E3/E5 candidate-cost tables; 5 evaluation splits | D satisfies the locked GO criterion; A/B/C are progressive ablations, not a clean factorial 2×2 | Fit calibrators and grids on fit-half only; held-out half evaluation only; cells and thresholds are correlated |
| D remains advantageous after real end-to-end overhead | [Router overhead FINAL LOCK](../reports/router_overhead_v1_20260922.md) and named E1/E3 measurement/replay JSON artifacts | 3 seeds × 2 structurally different backends; shared per-image predictions; 120 operating cells | 69 fair cells: 51 wins, 16 ties, 2 losses; 67/69 win-or-tie; macro ΔmIoU +0.0241 (2.41 points); D 0/120 and A 51/120 violating operating cells | Cross-backend deployment robustness, not independent accuracy replication; 120 cells are not independent samples; zero D violations partly follows from hard-budget design |
| Optimized E3 GPU risk computation matches the production reference | [Production audit script](../scripts/audit_gpu_risk_kernel_production.py), raw `reports/audit_gpu_risk_kernel_production_E3.json` | 40 real TensorRT outputs, 10 per level; production PyTorch risk function and real A/D policy code | Maximum scalar risk error 7.15e-07 nats versus 1e-4 tolerance; zero observed decision mismatches | Acceptance threshold is a correctness criterion, not a population-level statistical estimate of 99.9% agreement |
| Router overhead is architecture-dependent | [Router overhead FINAL LOCK](../reports/router_overhead_v1_20260922.md), [E3 harness](../scripts/measure_router_overhead.py), [E1 harness](../scripts/measure_router_overhead_hailo.py) | E3 TensorRT/CUDA and E1 Hailo-8 | E3 naive host risk evaluation can dominate latency and requires accelerator-resident optimization; E1 is dominated by mandatory network-group activation/deactivation | E1 warm semantics differ from E3; cold scenarios are not directly comparable; do not attribute all E3 host-path time to NumPy entropy without decomposition |

## Known limitations to preserve

- Probe-signal AURC is weakest under ACDC/night; avoid the unqualified word “reliable.”
- The two backend replays reuse model predictions.
- Breakpoint budgets can make device choices depend primarily on latency rank.
- The fair-cell filter must be explained and accompanied by all-cell budget-violation results.
- QAT uses fake quantization; compiled INT8 engine accuracy is not established.
- No calibrated external power-meter evidence; energy claims are excluded.
- Temporal routing, UIoU, cold/reload frontier replay and detailed E3 host-path decomposition are future work, not missing gating evidence.

## Numerical invariants for review

- Overall fair cells: `69/120`.
- W/T/L on fair cells: `51/16/2`.
- Win-or-tie: `67/69 = 97.1%`.
- Macro average: `+0.0241 mIoU = +2.41 mIoU points`.
- D violating operating cells: `0/120`.
- A violating operating cells: `51/120`.
