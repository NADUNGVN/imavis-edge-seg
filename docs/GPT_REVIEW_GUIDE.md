# GPT Review Guide — PACE-Seg

## Review target

This packet is for an adversarial scientific review of PACE-Seg before manuscript submission.

- Deployment-matched method commit: `ababda12a9bfb6a5f92a7d79aad3560d361f863f`
- Deployment-matched result commit: `b57fcb2`
- Manuscript snapshot tag: `ivc-deployment-matched-v4-20260930`
- Target venue: *Image and Vision Computing* (IMAVIS)
- Scope: scientific validity, novelty, leakage, numerical consistency, deployment validity, reproducibility, and claim boundaries.

Treat README statements as claims, not evidence. Trace every headline number to the report, raw JSON, and implementation. Operating cells and backend replays are correlated; do not treat them as independent statistical samples.

**Scientific correction completed.** The historical router fit-half feature used a
ground-truth-valid entropy mask unavailable at deployment. The final Run A--C package
reruns fitting and deployment with all-pixel entropy and replays measured complete
warm-route costs on E1 and E3. Historical masked-fit arithmetic remains auditable but
is not submission evidence.

## Required reading order

1. [README](../README.md) — project map and living status only.
2. [Research plan](RESEARCH_PLAN.md) — research questions, contribution targets, go/no-go rules and ablations.
3. [Canonical LaTeX manuscript](../paper/submission/ivc_2026-09-30_v4/main.tex), then
   consult the [historical Markdown draft](MANUSCRIPT_DRAFT.md) and
   [skeleton](MANUSCRIPT_SKELETON.md) only for provenance.
4. [Baseline comparison](../reports/baseline_comparison_gap_check_20260912.md) — RQ2 and three-seed near-parity.
5. [Pareto search](../reports/pareto_search_v1_20260912.md), [FLOPs baseline](../reports/flops_baseline_v1_20260917.md), and [RQ1 budget sweep](../reports/rq1_budget_sweep_v1_20260920.md).
6. [QAT journey](../reports/qat_v1_20260913.md), [failed max calibration](../reports/calibrated_qat_v1_20260917.md), and [QAT rescue screen](../reports/qat_rescue_2x2_screen_infra_20260920.md).
7. [Deployment-matched FINAL router report](../reports/router_deployment_matched_v1_20260930.md), [manifest](../reports/router_deployment_matched_20260929/manifest.json), [summary](../reports/router_deployment_matched_20260929/summary.json), and all six replay JSON files.
8. [Router progressive ablation](../reports/router_progressive_ablation_v1_20260921.md) and [router overhead report](../reports/router_overhead_v1_20260922.md) as historical/mechanistic support.
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

## Final router result to audit

The deployment-matched artifacts report 70 of 74 fair operating cells (56 wins,
14 ties, 4 losses), a macro-averaged gain of 0.0283549 mIoU, D 0/120 violating
cells, and A 46/120. Pooled D reports 58/14/2 on the same fair subset, +0.0295859
mIoU, and 0/120 violating cells. The reviewer must reproduce these values from the
six replay JSON files rather than accepting the report summary.

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

The strict review must verify that final router artifacts use identical all-pixel entropy at fit and deployment. It must then challenge: the fair-cell filter; fit-half/held-out separation; pooled-calibrator construction; canonical per-run risk grids; budget-matched operating-point selection; oracle use; run provenance; latency double counting; tiny-output reuse; CUDA synchronization; Hailo activation semantics; stale HEFs; QAT scope; statistical dependence; missing closest work; and stale/conflicting numbers across README, reports and manuscript.

The reviewer should recommend new experiments only when a central claim cannot stand without them. Otherwise separate mandatory manuscript corrections from optional extensions.
