# 02 — Critical review (methodology, novelty, deployment)

## Summary judgement
A careful, honest measurement study with a clear negative result: on AGX Xavier and Hailo-8, static deployment beats input-adaptive routing once routes and static engines are timed in the same harness. The evidence for the router ablation and for the static comparison is solid; the weakest points are (i) reporting of the static comparison denominator (04 R-1), (ii) image-level uncertainty without seed variance and with a correlated alternating-index split, (iii) unmeasured attribution of the Hailo-8 overhead, and (iv) a narrow candidate set and one probe, which limit generality.

## Methodology review
- Fit/held-out separation is enforced for calibrators, thresholds, risk grids and operating points (verified in `router_review_analyses.py::run_policy`). Strength.
- Baseline fairness: static deployment charged its own latency — the fairest comparison; route-cost-charged static also reported. Strength. Static engines on Hailo-8 run with a continuously activated network group while routes activate per call; this is the real cost of switching, but should be stated explicitly as a design assumption (MINOR).
- PACE-Large and Fast-SCNN share the recipe; RQ2 is fair.
- Replay vs closed loop: accuracy from cached predictions, costs from tables; on-device replay for Run A confirms (≤0.6 point change). Adequate, limitation stated.

## Deployment-system review
- Overheads (AGX 0.7–2.7 ms; Hailo-8 36–50 ms across four configurations) are measured; components are not (MISSING EVIDENCE).
- Budget grid includes budgets below the cheapest route → infeasible cells (04 R-1).
- No energy measurement (stated).
- Cold-start excluded (stated).

## Novelty review
See 07. The contribution is a measurement methodology and a bounded negative result, which IVC accepts if framed precisely; current framing is consistent.

## Prioritized concerns
1. MAJOR — Table 7/Fig. 7 static rows: denominator and violation claim (fixable without experiments).
2. MAJOR — Uncertainty ignores seed variance; split correlated (report per-seed deltas; optional sequence-grouped replay).
3. MAJOR — Generality: 4 candidates, 1 probe, 2 datasets; state as scope (already in Limitations).
4. MINOR — Overclaiming wording in Abstract (same-harness scope, "adds nothing").
5. MINOR — Hailo overhead attribution.
6. MINOR — Fig. 10 missing Small column; Fig. 7 two-question layout.
7. BLOCKER (administrative) — placeholders.
