# 12 — V21 revision candidate: changelog, figure comparison, verification, open issues

Branch `v21-revision`. V20 sources untouched in `paper/submission/ivc_2026-10-08_v20_from_v19/`; V21 in `paper/submission/ivc_2026-10-09_v21_revision/`. Diffs: `V20_to_V21_{main,supplement,references}.diff` in the V21 folder.

Legend: **[A]** verified against an artifact; **[E]** editorial (wording, no numerical content).

## Changelog by location
| Location | Change | Type |
|---|---|---|
| Abstract | same-harness scope; feasible-cell static numbers 3.0 / 8.2; 12.5% / 37.5% static-only cells; calibration CI wording; break-even counterfactual | [A] |
| Introduction RQ3 paragraph, contribution 3 | "matches"/"not necessary" → no detectable gain | [A] CI |
| §4.4 Evaluation protocol | two grids defined with denominators; route-infeasible budgets; feasible primary / all diagnostic; paired fixed-seed image bootstrap, no seed variance; counterfactual break-even | [A] code |
| §5.1 RQ1 | Hailo-8 tiny-route expansion not attributed to components | [E] (timing scope) |
| §5.2 RQ2 | 1.5-point band separated for Fast-SCNN and PACE-Large | [A] eval JSONs |
| Table 7 | (a) route-budget ablation, (b) static-own-cost: feasible (primary), all (diagnostic), infeasible counts; caption | [A] phaseA JSON |
| Fig. 7 | two panels from `plot_fig7_v21.py` | [A] |
| §5.3 ablation | interval wording; no equivalence claim | [A] |
| §5.3 static | feasible primary −3.00 / −8.18 (UINT8 −10.97), T-hard −2.99 / −8.17, diagnostic −2.62 / −7.41, infeasible 15 / 45 | [A] |
| §5.3 break-even | counterfactual; bootstrap 0.6–1.5 / 2.1–5.2 ms | [A] phaseB JSON |
| Fig. 5 | y-axis "Candidate-only mean latency"; caption | [E] |
| Fig. 8 | bars labelled "static"/"route"; caption: components not timed | [E] |
| Fig. 9 | x-axis "Counterfactual routing overhead"; caption: measured operating point, no implementation measured | [E] |
| Fig. 10 | Small column added; route-class counts printed above grid; selection rule unchanged | [A] selection JSON |
| §6.1, §6.4, §6.5, Conclusion | Hailo overhead wording; route-budget scope of zero violations; per-seed and sequence-split robustness sentences | [A] |
| references.bib | AutoSegEdge authors (3, per Crossref), article number; Dynamic Routing pages 8550–8559 + DOI; MESS DOI, LNCS 13682 | [A] Crossref |
| Supplement §1, §3, Table S3 | two grids; feasible/all/infeasible columns with CIs | [A] |
| Supplement §8 (new), Table S7 | per-seed + sequence-split robustness; break-even bootstrap | [A] phaseB JSON |

## Figure comparison (V20 → V21)
| Fig. | V20 | V21 |
|---|---|---|
| 5 | y "Measured mean latency" | y "Candidate-only mean latency"; caption defines candidate-only |
| 7 | single forest, static rows over all 120 cells incl. infeasible | (a) ablation; (b) feasible (filled, primary) vs all (open, diagnostic), n = 105/75/120, axis to −9.5 |
| 8 | "S"/"C" small grey labels | "static"/"route" 6 pt ink labels |
| 9 | x "Routing overhead per frame" | x "Counterfactual routing overhead per frame" |
| 10 | Tiny/Medium/Large/Routed | Tiny/Small/Medium/Large/Routed + selection counts (0/27/61/365) |
V20 figures remain in `paper/submission/ivc_2026-10-08_v20_from_v19/figures/` and `paper/figures/generated/` for side-by-side comparison.

## Statistical tables (updated)
Table 7 (main), Table S3 (Hailo/AGX configurations with feasible CIs), Table S7 (per seed, pooled, sequence split). Sources: `reports/phaseA_static_feasibility_20261004.json`, `reports/phaseB_per_seed_20261004.json`, `reports/phaseB_sequence_split_20261004.json`, `reports/phaseB_breakeven_bootstrap_20261004.json`.

## Verification
- `scripts/verify_v21_numbers.py`: **28/28 PASS** (Table 7 rows, CIs, counts, Abstract/Results sentences, retired phrases absent).
- Phase A bootstrap re-run: byte-identical JSON.
- Builds: main 16 pp, supplement 12 pp; no LaTeX errors or undefined references (`build_main.log`, `build_supplement.log`).
- Visual inspection performed on main pp. 1–2, 8–14 and supplement pp. 8–9 at render resolution; remaining pages were inspected in V20 and changed only in text.

## Unresolved scientific issues
1. Hailo-8 route overhead components not measured (needs device profiling — not approved).
2. Compiled-engine accuracy and on-device replay only for Run A (needs device time — not approved).
3. Training-seed variability not in any interval; 3 seeds give a descriptive range only.
4. Sequence-aware split is coarse for night (2 recordings) and fog (3).
5. Risk target is per-image pixel error; class-balanced alternatives not evaluated (optional replay).
6. Break-even interpolation on a 7-point α grid; interval excludes discretization error.
7. RADLER reference not identified (only a radar-detection RADLER found).
8. Remaining bib entries not individually verified online (ENet, ERFNet, BiSeNet, etc.); format inconsistent (initials vs full names).
9. Accuracy–latency frontier with static and routed points under one cost definition: feasible from existing data (`router_mean_budget_frontier`, same-harness tables) — **proposed only, not built**.

## Submission-readiness checklist (not ready)
- [ ] Author metadata, affiliation, e-mail, funding, repository/DOI, acknowledgements (author input)
- [ ] AI-use declaration lists every tool used (author input)
- [x] Table 7 / Fig. 7 denominators and violation scope corrected
- [x] Wording: calibration CI, break-even counterfactual, Hailo attribution, 1.5-point band, same-harness scope
- [x] Figures 5, 8, 9, 10 revised
- [x] Robustness: per seed, sequence split, break-even CI (supplement)
- [ ] Bibliography fully verified and normalized
- [ ] Overleaf build and final page-by-page proof of the V21 PDF
- [ ] Decision on items 1–2 (device time)
