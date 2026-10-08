# 01 — Manuscript inventory and provenance (V20, 2026-10-08)

Status labels used in all audit files: **VERIFIED ERROR**, **SUSPECTED ISSUE**, **MISSING EVIDENCE**, **SUGGESTED IMPROVEMENT**.

## Scope and method
- Audited source: `paper/submission/ivc_2026-10-08_v20_from_v19/` (main.tex, supplement.tex, references.bib, figures/, tables/), commit `3f85a41`.
- Numbers were traced to the JSON artifacts listed below by re-reading them with Python; one new read-only check was run (`scripts/audit_static_cells.py` → `reports/audit_static_cells_20261004.json`). No training, benchmarking, or device measurement was run.
- The skills named in the brief (`peer-review`, `scientific-critical-thinking`, …) are **not installed** in this environment; equivalent checks were done manually. No content was sent to external review/editing services. Web access was used only for bibliographic look-ups.

## Artifacts
| Category | Location | Notes |
|---|---|---|
| Manuscript | `main.tex` (≈510 lines), `supplement.tex` | elsarticle 5p; supplement elsarticle review |
| Bibliography | `references.bib` (54 entries) | mixed author formats (initials vs full) |
| Main figures (10) | `figures/fig1_teaser_v20.pdf` … `fig10_stratified_v20.pdf` | generators below |
| Figure scripts | `paper/figures/scripts/{render_qualitative_v20,plot_v20_final,plot_route_static_v20,plot_flops_latency_v20,plot_v20_3d}.py`, `paper/figures/html/*.html` + `render_html_figs.py` | all re-runnable locally |
| Per-image router dumps | `reports/landscape_20261004/run_{a,b,c}_{per_image,evaluation}.json` | seeds 0–2, checkpoint SHA in `_metadata` |
| RQ2 evals | `reports/landscape_20261004/eval_{supernet,pace_large,fast_scnn}_seed{0,1,2}.json` | full validation sets |
| FLOPs | `reports/landscape_20261004/flops_landscape.json` | |
| Candidate-only latency LUT | `outputs/benchmark_lookup_table_landscape.csv` | 4 devices |
| Same-harness route/static | `reports/static_vs_route_E3_20261004.json`, `static_vs_route_E1_{explicit,scheduler}_{float32,uint8}_20261004.json` | 50 warm-up / 500 timed |
| Router analyses | `reports/router_review_analyses_20261004.json`, `_v2_20261004.json`, `router_same_harness_analysis_20261004.json`, `router_breakeven_20261004.json`, `router_nested_bootstrap_20261004.json`, `router_by_condition_20261004.json` | |
| Compiled-engine accuracy | `reports/compiled_eval_E{1,3}_20261004.json`, `router_ondevice_replay_20261004.json` | **Run A only** |
| Qualitative assets | `reports/qualitative_assets_v20{,_stratified}/` + selection JSONs | audited against confusion matrices |
| Checkpoints | server only (`outputs/pace_seg_landscape_*`) | not in repo (expected) |
| Timing scripts | `scripts/measure_static_vs_route_{trt,hailo}.py` | no per-component timers (see 04) |

## Provenance map (headline numbers)
| Manuscript claim | Value | Artifact | Status |
|---|---|---|---|
| FLOPs range | 110.2× | flops_landscape.json | verified |
| Measured latency range | 12.3–18.1× | benchmark_lookup_table_landscape.csv (via plot_flops_latency_v20.py) | verified |
| Shared − alone (CS, fog, night, rain, snow) | −0.94, −1.74, −2.75, −1.61, −1.71 | eval_supernet / eval_pace_large | verified |
| Elastic − Fast-SCNN | +0.75, −1.80, −0.77, −3.95, −1.38 | eval_supernet / eval_fast_scnn | verified |
| D − A (fair) | +2.87 [2.52, 3.26], nested [1.85, 3.28], 54/6/2 of 62 | router_review_analyses / nested_bootstrap | verified |
| D − A-hard | +1.41 [1.19, 1.63], nested [0.99, 1.70], 48/72/0 | same | verified |
| D − T-hard | −0.03 [−0.11, 0.09], 16/80/24 | _v2 | verified |
| D − static (own cost) AGX / Hailo-8 | −2.62 (0/61/59) / −7.41 (0/21/99) | router_same_harness_analysis | verified as computed; **denominator includes infeasible cells — see 04 R-1** |
| Break-even | 1.3 ms / 4.3 ms | router_breakeven + plot script | verified (counterfactual) |
| Compiled TRT / Hailo deltas | ±0.02 / up to −1.64 (tiny), ≤0.52 (large) | compiled_eval_* | verified (Run A only) |
| Hailo entropy rank corr. ≥0.88 | 0.88 (night) | compiled_eval_E1 entropy | verified |

## Missing evidence
- **MISSING EVIDENCE:** per-component timing of the Hailo-8 route (probe inference, host entropy, activation/deactivation, transfer, selected inference). The scripts time only the full route and the static engine.
- **MISSING EVIDENCE:** compiled-engine accuracy and on-device replay for Runs B and C.
- **MISSING EVIDENCE:** author metadata (affiliation, e-mail, funding, repository/DOI, acknowledgements).
