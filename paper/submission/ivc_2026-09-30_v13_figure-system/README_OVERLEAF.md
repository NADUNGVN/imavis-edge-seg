# PACE-Seg Overleaf package, unified figure system v13

This folder contains a self-contained `elsarticle` main manuscript and independent
supplement for *Image and Vision Computing*, Special Issue: Complex Environment
Vision. Compile `main.tex` for the 25-page paper and `supplement.tex` for the
11-page supplementary document. The package requires no shell escape.

## Upload to Overleaf

1. Upload `PACE_SEG_OVERLEAF_PACKAGE_V13_20260930.zip` as a new project, or upload
   `main.tex` plus the complete `figures/` and `tables/` folders to the existing project.
2. Select `main.tex` as the main document.
3. Compile with pdfLaTeX.
4. To export the supplement, select `supplement.tex` as the main document and compile again.

The repository also provides `paper/scripts/build_manuscript_pdf.ps1`. Running it
without arguments updates the stable main-paper artifact at
`output/pdf/PACE-Seg_IVC_Manuscript.pdf` from this package's `main.tex`.
Use this command after every TeX or figure change so the checked-in review PDF and
the Overleaf source cannot drift:

```powershell
.\paper\scripts\build_manuscript_pdf.ps1
.\paper\scripts\build_manuscript_pdf.ps1 -TexFile supplement.tex `
  -OutputName PACE-Seg_IVC_Supplement_V13_20260930.pdf
```

`MAIN_TO_SUPPLEMENT.md` lists every block moved out of the V11 main paper. Headline
numbers and claims are unchanged.

## Main-paper visual narrative

| Figure | Scientific question | Reproducible source |
|---|---|---|
| `fig1_data_gt_routed.png` | How do clean/adverse inputs, ground truth, and routed outputs relate? | `paper/figures/scripts/assemble_visual_story_v13.py` |
| `fig2_candidate_family.pdf` | How does the active encoder-decoder expose four static width-depth-resolution capacities and backend builds? | `paper/figures/scripts/render_fig2_architecture_v13.py` |
| `fig3_condition_probe_routes.pdf` | How do candidate-specific risk and measured cost combine conceptually before static-engine selection? | `paper/figures/v13/tikz/fig3_conceptual_pipeline_v13.tex` |
| `fig4_input_output_pipeline.pdf` | How does policy D combine candidate risk and complete route cost in one audited decision? | `paper/figures/scripts/plot_complete_routing_decision_v11.py` |
| `fig5_flops_vs_latency.pdf` | Why is proportional FLOPs insufficient for a latency budget? | `paper/figures/scripts/plot_flops_latency_v13.py` |
| `fig6_rq2_near_parity.pdf` | Does shared elastic training preserve matched-capacity segmentation quality? | `paper/figures/scripts/plot_rq2_near_parity_v13.py` |
| `fig7_routing_decision_example.pdf` | How can one image and risk vector yield different feasible decisions under the same budget? | `paper/figures/scripts/plot_routing_decision_example_v13.py` |
| `fig8_router_deployment_matched.pdf` | Does candidate-specific hard-budget routing improve quality under measured complete route costs? | `paper/figures/scripts/plot_router_deployment_matched_v13.py` |
| `fig9_qualitative_grid.png` | How do candidate capacity, selected output, and error differ on held-out images? | `paper/figures/scripts/assemble_visual_story_v13.py` |
| `figS1_failure_gallery.png` | What errors appear across the six frozen audited examples? | `paper/figures/scripts/assemble_visual_story_v13.py` |
| `figS2_route_cost_expansion.pdf` | How much larger is the complete routing path than isolated candidate inference? | `paper/figures/scripts/plot_route_cost_expansion.py` |
| `figS3_additional_qualitative.png` | How do small, large, and routed outputs compare on the same six audited examples? | `paper/figures/scripts/assemble_visual_story_v11.py` |
| `figS4_local_crops.png` | What local errors are visible under one fixed crop rule? | `paper/figures/scripts/assemble_visual_story_v11.py` |

The raster panels are deterministic assemblies of real dataset inputs, ground truth,
and Run A predictions. Their IDs, hashes, decisions, and exact confusion-matrix audit
are recorded in `paper/figures/qualitative/generated/render_manifest.json`; the V13
rearrangement audit is `paper/figures/generated/visual_story_v13_audit.json`. Figures
S3 and S4 retain the V11 rearrangement of the same six frozen examples and do not add
easy/hard samples.

Figure 2 retains the raw, exact, and publication Architecture IR under
`paper/figures/v13/architecture/`; its renderer reads `paper/tables/candidate_family.csv`
and verifies the active implementation sources. All numerical plots parse canonical
JSON/CSV artifacts and do not embed experimental outcomes manually.

No generative or manually retouched imagery is used as experimental evidence. Raw
datasets and checkpoints are not distributed in this package.

## Evidence boundaries retained in the manuscript

- E1 and E3 reuse segmentation predictions and are not independent accuracy replications.
- Operating cells are correlated; 74 fair cells are conditionally defined by A feasibility.
- D's zero violations are partly structural under the hard budget.
- Hardware accuracy is an offline replay using measured complete warm-route costs.
- QAT uses fake quantization and does not establish compiled INT8 accuracy.
- No energy result is claimed.
- Pooled D is reported alongside configured condition-specific D.

## Repository references

- Deployment-matched method commit: `ababda12a9bfb6a5f92a7d79aad3560d361f863f`
- Deployment-matched result commit: `b57fcb2`
- Parent manuscript snapshot tag: `ivc-expanded-v7-20260930`
- Previous visual-scientific release: `ivc-visual-scientific-v3-20260929`
- Parent visual snapshot tag: `ivc-early-visuals-v8-20260930`
- Parent figure revision: tag `ivc-figure-revision-v9-20260930`
- Parent visual snapshot: tag `ivc-research-visuals-v10-20260930`
- Parent visual snapshot: V11 final figure map
- Parent compressed manuscript: V12
- Revision v13 source: this folder, `paper/figures/`, and `paper/tables/`
