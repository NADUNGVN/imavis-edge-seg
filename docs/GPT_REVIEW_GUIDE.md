# GPT Review Guide — PACE-Seg

## Review target

This packet is for an adversarial scientific review of PACE-Seg before manuscript submission.

- Evidence baseline commit: `1c4d42a7ea525931154ebb2f9015d898ef7771a5`
- Review snapshot/tag: `gpt-review-v1-20260929`
- Target venue: *Image and Vision Computing* (IMAVIS)
- Scope: scientific validity, novelty, leakage, numerical consistency, deployment validity, reproducibility, and claim boundaries.

Treat README statements as claims, not evidence. Trace every headline number to the report, raw JSON, and implementation. Operating cells and backend replays are correlated; do not treat them as independent statistical samples.

**Scientific correction after the frozen review snapshot.** The historical router
fit-half feature used a ground-truth-valid entropy mask unavailable at deployment.
The evaluator now uses all-pixel entropy in both phases, but the three checkpoints
and licensed datasets must be rerun. Therefore the router arithmetic below is an
audit invariant for the historical artifacts, not a submission headline.

## Required reading order

1. [README](../README.md) — project map and living status only.
2. [Research plan](RESEARCH_PLAN.md) — research questions, contribution targets, go/no-go rules and ablations.
3. [Manuscript draft](MANUSCRIPT_DRAFT.md) and [skeleton](MANUSCRIPT_SKELETON.md).
4. [Baseline comparison](../reports/baseline_comparison_gap_check_20260912.md) — RQ2 and three-seed near-parity.
5. [Pareto search](../reports/pareto_search_v1_20260912.md), [FLOPs baseline](../reports/flops_baseline_v1_20260917.md), and [RQ1 budget sweep](../reports/rq1_budget_sweep_v1_20260920.md).
6. [QAT journey](../reports/qat_v1_20260913.md), [failed max calibration](../reports/calibrated_qat_v1_20260917.md), and [QAT rescue screen](../reports/qat_rescue_2x2_screen_infra_20260920.md).
7. [Router progressive ablation](../reports/router_progressive_ablation_v1_20260921.md) and canonical per-device/per-seed JSON files named in that report.
8. [Router overhead and FINAL LOCK](../reports/router_overhead_v1_20260922.md), including E1/E3 raw measurement, replay, and audit JSON files named there.
9. Implementation:
   - [policy](../src/imavis_edge_seg/router/policy.py)
   - [risk probe](../src/imavis_edge_seg/router/risk_probe.py)
   - [calibrator](../src/imavis_edge_seg/router/calibrator.py)
   - [router evaluation](../scripts/evaluate_router.py)
   - [overhead replay](../scripts/replay_router_with_overhead.py)
   - [TensorRT overhead harness](../scripts/measure_router_overhead.py)
   - [Hailo overhead harness](../scripts/measure_router_overhead_hailo.py)
   - [production CUDA audit](../scripts/audit_gpu_risk_kernel_production.py)
   - [quantization](../src/imavis_edge_seg/training/quantization.py)
10. [Claims–evidence matrix](CLAIMS_EVIDENCE_MATRIX.md), then use [strict review prompt](GPT_STRICT_REVIEW_PROMPT.md).

## Historical router result to audit

The old artifacts report 67 of 69 fair operating cells (51 wins, 16 ties, 2 losses), a macro-averaged gain of 0.0241 mIoU, D 0/120 violating cells, and A 51/120. These values must remain reproducible for audit, but they cannot support the final RQ3 claim until the deployment-matched Run A--C rerun is complete.

Required boundary: the two backend evaluations share model predictions and demonstrate cross-backend deployment robustness, not independent accuracy replication.

## Claims that must not be made

- Do not treat 120 operating cells as 120 independent samples.
- Do not describe zero budget violations as general reliability; D enforces a hard budget by construction.
- Do not call the router universally reliable, especially under ACDC/night where probe AURC is weakest.
- Do not claim compiled-engine INT8 accuracy from fake-quant QAT evidence.
- Do not claim the two hardware backends provide independent accuracy replication.
- Do not claim routing decisions always differ by device; breakpoint budgets can reduce decisions to latency-rank comparisons.
- Do not substitute FLOPs for measured latency or claim energy results without a power meter.

## Reviewer focus

The strict review must first verify that final router artifacts use identical all-pixel entropy at fit and deployment. It must then challenge: the fair-cell filter; fit-half/held-out separation; canonical per-run risk grids; budget-matched operating-point selection; oracle use; run provenance; latency double counting; tiny-output reuse; CUDA synchronization; Hailo activation semantics; stale HEFs; QAT scope; statistical dependence; missing closest work; and stale/conflicting numbers across README, reports and manuscript.

The reviewer should recommend new experiments only when a central claim cannot stand without them. Otherwise separate mandatory manuscript corrections from optional extensions.
