# PACE-Seg Overleaf package, final figure map v11

This folder is a self-contained `elsarticle` manuscript for *Image and Vision
Computing*, Special Issue: Complex Environment Vision. Compile `main.tex` with
pdfLaTeX. The package contains vector diagrams/plots, audited raster panels, and a
generated supplementary table;
it requires no shell escape.

## Upload to Overleaf

1. Upload `PACE_SEG_OVERLEAF_PACKAGE_V11_20260930.zip` as a new project, or upload
   `main.tex` plus the complete `figures/` and `tables/` folders to the existing project.
2. Select `main.tex` as the main document.
3. Compile with pdfLaTeX.

The repository also provides `paper/scripts/build_manuscript_pdf.ps1`. Running it
from the repository updates the stable review artifact at
`output/pdf/PACE-Seg_IVC_Manuscript.pdf` from this package's `main.tex`.
Use this command after every TeX or figure change so the checked-in review PDF and
the Overleaf source cannot drift:

```powershell
.\paper\scripts\build_manuscript_pdf.ps1
```

## Main-paper visual narrative

| Figure | Scientific question | Reproducible source |
|---|---|---|
| `fig1_data_gt_routed.png` | How do clean/adverse inputs, ground truth, and routed outputs relate? | `paper/figures/scripts/assemble_visual_story_v11.py` |
| `fig2_candidate_family.pdf` | How do four capacities from one shared family grow in resolution, width, depth, compute, and measured candidate latency? | `paper/figures/scripts/plot_candidate_family_silhouettes_v11.py` |
| `fig3_condition_probe_routes.pdf` | How does one image move from preparation and probe inference to calibrated risks, feasibility, and selected output? | `paper/figures/scripts/plot_visual_routing_pipeline_v11.py` |
| `fig4_input_output_pipeline.pdf` | How does policy D combine candidate risk and complete route cost in one audited decision? | `paper/figures/scripts/plot_complete_routing_decision_v11.py` |
| `fig5_flops_vs_latency.pdf` | Why is proportional FLOPs insufficient for a latency budget? | `paper/figures/scripts/plot_flops_latency_v11.py` |
| `fig6_rq2_near_parity.pdf` | Does shared elastic training preserve matched-capacity segmentation quality? | `paper/figures/scripts/plot_rq2_near_parity_v11.py` |
| `fig7_routing_decision_example.pdf` | How can one image and risk vector yield different feasible decisions under the same budget? | `paper/figures/scripts/plot_routing_decision_example_v11.py` |
| `fig8_router_deployment_matched.pdf` | Does candidate-specific hard-budget routing improve quality under measured complete route costs? | `paper/figures/scripts/plot_router_deployment_matched_v11.py` |
| `fig9_qualitative_grid.png` | How do candidate capacity, selected output, and error differ on held-out images? | `paper/figures/scripts/assemble_visual_story_v11.py` |
| `figS1_failure_gallery.png` | What errors appear across the six frozen audited examples? | `paper/figures/scripts/assemble_visual_story_v11.py` |
| `figS2_route_cost_expansion.pdf` | How much larger is the complete routing path than isolated candidate inference? | `paper/figures/scripts/plot_route_cost_expansion.py` |
| `figS3_additional_qualitative.png` | How do small, large, and routed outputs compare on the same six audited examples? | `paper/figures/scripts/assemble_visual_story_v11.py` |
| `figS4_local_crops.png` | What local errors are visible under one fixed crop rule? | `paper/figures/scripts/assemble_visual_story_v11.py` |

The raster panels are deterministic assemblies of real dataset inputs, ground truth,
and Run A predictions. Their IDs, hashes, decisions, and exact confusion-matrix audit
are recorded in `paper/figures/qualitative/generated/render_manifest.json`; the V11
rearrangement audit is `paper/figures/generated/visual_story_v11_audit.json`. Figures
S3 and S4 reorganize the same six frozen examples and do not add easy/hard samples.

Figure 2 retains editable Archify source and validation receipts under
`paper/figures/archify/fig2_candidate_family_v11/`. Its publication silhouette is
generated from `paper/tables/candidate_family.csv`; all numerical plots parse canonical
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
- Revision v11 source: this folder, `paper/figures/`, and `paper/tables/`
