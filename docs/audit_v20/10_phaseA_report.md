# 10 — Phase A report (V21 revision, branch `v21-revision`)

## A1. Why budgets below the cheapest route are route-infeasible (verified in code)
- `router_same_harness_analysis.configurations` → for each device, route table $C_h$ and static table $S_h$ from the same `static_vs_route_*_20261004.json`.
- Static-own-cost budgets = `sorted(set(C_h) ∪ set(S_h))` → 8 budgets.
- Every route starts with the tiny probe, so $\min_\ell C_h(\ell) = C_h(\text{tiny})$. Since $S_h(\text{tiny}) < C_h(\text{tiny})$ on both devices (1.18 < 1.84 ms; 6.27 < 44.26 ms), budgets $< C_h(\text{tiny})$ admit the static tiny engine but no route.
- In `router_review_analyses.d_choice`, an empty feasible set returns `order[0]` (tiny); `run_policy` records `violation > 0`. `static_fair` returns a static candidate whenever $S_h(\ell) \le B$.
- Route-infeasible budgets: AGX Xavier 1.18 ms; Hailo-8 (explicit FLOAT32) 6.27, 12.07, 30.91 ms.

## A2–A4. Recomputed results (`scripts/phaseA_static_feasibility.py` → `reports/phaseA_static_feasibility_20261004.json`)
Static-own-cost grid per backend: 3 runs × 5 splits × 8 budgets = 120 cells.

| Backend | Cells | W/T/L | Mean Δ (D − static) | 95% CI (fresh) | Role |
|---|---|---|---|---|---|
| AGX Xavier | 105 | 0/46/59 | −3.00 | [−3.07, −2.72] | primary (route-feasible) |
| AGX Xavier | 120 | 0/61/59 | −2.62 | [−2.69, −2.38] | diagnostic (all) |
| AGX Xavier | 15/120 (12.5%) | — | — | — | static feasible, route infeasible; D violates in all 15 |
| Hailo-8 FLOAT32 | 75 | 0/6/69 | −8.18 | [−8.42, −7.40] | primary |
| Hailo-8 FLOAT32 | 120 | 0/21/99 | −7.41 | [−7.61, −6.76] | diagnostic |
| Hailo-8 FLOAT32 | 45/120 (37.5%) | — | — | — | static feasible, route infeasible; D violates in all 45 |
| Hailo-8 UINT8 | 75 / 120 | 0/6/69 / 0/21/99 | −10.97 / −9.15 | [−11.26, −9.94] / [−9.39, −8.33] | primary / diagnostic |
| T-hard, AGX / Hailo-8 | 105 / 75 | 1/49/55 / 1/9/65 | −2.99 / −8.17 | [−3.09, −2.71] / [−8.39, −7.40] | primary |

D has 0 violations on route-feasible cells of both grids. Route-budget grid (3 runs × 2 backends × 5 splits × 4 budgets = 120): D, A-hard, T-hard, pooled D, LOCO D 0 violations at median/p95/p99 (verified from `router_review_analyses_20261004.json`).

Shares 12.5% / 37.5% are shares of the evaluated budget grid, not deployment deadline-miss rates.

## Bootstrap verification
| Question | Answer | Evidence |
|---|---|---|
| Paired between D and static? | Yes. Each replicate draws one index vector per split; both policies' per-image confusion matrices are indexed with it (`router_review_analyses_v2.bootstrap`). | code |
| Structure across runs/budgets preserved? | Yes. The same per-split resample is applied to every run, backend and budget cell of that split. | code |
| Training-seed variability included? | **No.** Image sampling only; now stated in Table 7 caption, Fig. 7 caption, §4.4. | text |
| Reproducible with fixed seed? | **Yes.** Full re-run (`PHASEA_OUT` to scratch) produced a byte-identical JSON (`identical`). | rerun |
| All-cell vs feasible computed on the right subsets? | Yes. `feasible = budget ≥ C_h(tiny)`; counts 105/75 and 15/45 match budget lists. | JSON |
| Not claimed: a more negative CI after filtering = stronger significance. | Stated explicitly in §5.3 ("two subsets differ in their budgets… should not be read as stronger evidence"). | text |

## Prose and captions changed (A5–A9)
| Location | Change | Basis |
|---|---|---|
| Abstract | same-harness scope = two primary backends; candidate latency on four devices | artifact scope |
| Abstract | "adds nothing" → no detectable gain, CI excludes > 0.09 | CI [−0.11, 0.09] |
| Abstract | static advantage 3.0 / 8.2 (route-feasible) + 12.5% / 37.5% static-only cells | phaseA JSON |
| Abstract, §4.4, §5.3, Fig. 9 caption, Discussion, Conclusion | break-even labelled counterfactual; no optimized implementation measured | method |
| Introduction | "matches" → "no detectable gain"; contribution 3 rephrased | CI |
| §4.4 | two grids defined with denominators; route-infeasible budgets explained; feasible = primary, all = diagnostic; bootstrap paired, fixed seed, image-only | code + JSON |
| §5.1 | Hailo-8 tiny-route expansion: components not timed separately (no attribution) | timing scripts |
| §5.2 | 1.5-point band: Fast-SCNN vs PACE-Large separated | eval JSONs |
| Table 7 | split (a) route-budget ablation / (b) static-own-cost with primary, diagnostic, infeasible rows; caption rewritten | phaseA JSON |
| Fig. 7 | two panels; (b) filled = feasible primary, open = all-cell diagnostic; n shown; axis to −9.5 | `plot_fig7_v21.py` |
| §5.3 static paragraph | rewritten with feasible primary, diagnostic, infeasible counts, T-hard | phaseA JSON |
| §5.3 ablation | "statistically indistinguishable" → interval statement, no equivalence | CI |
| §6.1, §6.4 | Hailo overhead not attributed to activation alone; per-frame budget below cheapest route | text |
| §6.5 Measurement | zero-violation scope = route-budget grid; Hailo components not timed | text |
| Conclusion | feasible-budget numbers; counterfactual; calibration wording | phaseA JSON |
| Supplement §1, §3, Table S3 | two grids; feasible / all / infeasible columns with CIs | phaseA JSON |

## Build and checks
- `tectonic` build of main.tex and supplement.tex: no errors, no undefined references (logs `build_main.log`, `build_supplement.log` in the V21 folder).
- `scripts/verify_v21_numbers.py`: 28/28 checks pass (table rows, CIs, counts, abstract/result sentences, absence of retired phrases).
- Visual inspection: pages 1–2, 8–14 of the main PDF inspected at render resolution (Table 7 p.10, Fig. 7 p.11).
