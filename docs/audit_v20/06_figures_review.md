# 06 — Figure-by-figure review

All figures were inspected as rendered PNG/PDF in this session; inspection "at final column width" was done on the compiled draft PDF pages, not on a printed proof.

| Fig. | Purpose | Source / script | Findings | Proposed change |
|---|---|---|---|---|
| 1 Teaser | input- and device-dependent choice; overhead | render_qualitative_v20.py `sample_teaser`; costs from router_same_harness_analysis | OK; panel (c) shows only Hailo-8 large. Text sizes now 7 pt. | Optionally add AGX bar for contrast. |
| 2 Pipeline (HTML) | inference flow | paper/figures/html/fig2_pipeline.html | Arrows/labels fixed; uses Run A rain image. Policy text small at column width (~5.5 pt). | Increase policy/legend text 1 pt. |
| 3 Architecture (HTML) | elastic axes, candidates | fig3_architecture.html, candidate_family_landscape.csv | Block heights are schematic (not to scale) — not stated. | Add "schematic, not to scale" to caption. |
| 4 Worked decision | one D decision | `sample_decision` | Colorbar present; risk values match text. | — |
| 5 FLOPs vs latency | RQ1 | plot_flops_latency_v20.py, LUT | Y axis says "Measured mean latency"; that it is **candidate-only** latency is only in the caption. | Axis label "candidate-only mean latency (ms)". |
| 6 RQ2 dumbbell | shared vs alone | plot_v20_final.py `fig_rq2`, eval_*.json | Deltas correct (−0.9, −1.7, −2.8, −1.6, −1.7). | Pair with text fix 04 R-2. |
| 7 Forest | ablation + static | plot_v20_final.py `fig_forest` | Mixes two questions (router ablation; routing vs static) on one axis; static rows include infeasible cells (R-1). | Two panels: (a) router ablation, (b) static comparisons over feasible cells. |
| 8 Route vs static | overhead | plot_route_static_v20.py | Labels S/C below bars small; hatch good. No component breakdown (not measured). | Do not add breakdown without data. |
| 9 Break-even | counterfactual | plot_route_static_v20.py | Counterfactual nature only in caption. | Axis label "counterfactual overhead (ms)"; mark measured point "measured". |
| 10 Stratified | adaptivity | `sample_stratified` | Columns Tiny/Medium/Large/Routed — **no Small prediction column**, although one row is routed to Small; selection counts (0/27/61/365) only in caption; panels small. | Add Small column; add a small bar of route-class frequencies; enlarge to full page width with 3 rows. |

Accessibility: capacity palette is Okabe-Ito (colour-blind safe), Fig. 8 uses hatch, Fig. 9 uses sign + shading. Fonts: all ≥5.5 pt at print width; Fig. 2/3 smallest text is the weakest point.
