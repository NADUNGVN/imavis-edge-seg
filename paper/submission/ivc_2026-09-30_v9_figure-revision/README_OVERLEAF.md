# PACE-Seg Overleaf package, figure revision v9

This folder is a self-contained `elsarticle` manuscript for *Image and Vision
Computing*, Special Issue: Complex Environment Vision. Compile `main.tex` with
pdfLaTeX. The package contains vector diagrams/plots, audited raster panels, and a
generated supplementary table;
it requires no shell escape.

## Upload to Overleaf

1. Upload `PACE_SEG_OVERLEAF_PACKAGE_V9_20260930.zip` as a new project, or upload
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
| `fig1_data_gt_routed.png` | How do clean/adverse inputs, ground truth, and routed outputs relate? | `paper/figures/scripts/assemble_visual_story_v9.py` |
| `fig2_candidate_family.pdf` | How do four related capacities differ, and how are they deployed as static engines? | `paper/figures/archify/fig2_candidate_family_v9/candidate.json` |
| `fig3_condition_probe_routes.png` | How do raw inputs, tiny-probe outputs, and selected outputs relate across conditions? | `paper/figures/scripts/assemble_visual_story_v9.py` |
| `fig4_input_output_pipeline.pdf` | How does one real image pass from preprocessing through probe, risk calibration, budget filtering, and selected output? | `paper/figures/scripts/plot_input_output_pipeline.py` |
| `fig5_flops_vs_latency.pdf` | Why is proportional FLOPs insufficient for a latency budget? | `paper/figures/scripts/plot_flops_latency.py` |
| `fig6_rq2_near_parity.pdf` | Does shared elastic training preserve matched-capacity segmentation quality? | `paper/figures/scripts/plot_rq2_near_parity.py` |
| `fig7_routing_decision_example.pdf` | How can one image and risk vector yield different feasible decisions under the same budget? | `paper/figures/scripts/plot_routing_decision_example.py` |
| `fig8_router_deployment_matched.pdf` | Does candidate-specific hard-budget routing improve quality under measured complete route costs? | `paper/figures/scripts/plot_router_deployment_matched.py` |
| `fig9_qualitative_grid.png` | How do the four capacities and the selected output differ on held-out images? | `paper/figures/scripts/render_qualitative_examples.py` |
| `figS1_failure_gallery.png` | What errors appear in deterministic median cases and the predetermined highest-error ACDC case? | `paper/figures/scripts/assemble_visual_story_v9.py` |
| `figS2_route_cost_expansion.pdf` | How much larger is the complete routing path than isolated candidate inference? | `paper/figures/scripts/plot_route_cost_expansion.py` |

The raster panels are deterministic assemblies of real dataset inputs, ground truth,
and Run A predictions. Their IDs, hashes, decisions, and exact confusion-matrix audit
are recorded in `paper/figures/qualitative/generated/render_manifest.json`; the V9
rearrangement audit is `paper/figures/generated/visual_story_v9_audit.json`.

The diagrams retain editable Archify JSON and exported SVG in the repository. The
data plots parse canonical JSON/CSV artifacts and do not embed experimental numbers
manually.

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
- Revision v9 source: this folder, `paper/figures/`, and `paper/tables/`
