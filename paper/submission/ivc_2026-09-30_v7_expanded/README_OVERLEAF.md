# PACE-Seg Overleaf package, visual-evidence revision v7

This folder is a self-contained `elsarticle` manuscript for *Image and Vision
Computing*, Special Issue: Complex Environment Vision. Compile `main.tex` with
pdfLaTeX. The package contains vector diagrams/plots, audited raster panels, and a
generated supplementary table;
it requires no shell escape.

## Upload to Overleaf

1. Upload `PACE_SEG_OVERLEAF_PACKAGE_V7_20260930.zip` as a new project, or upload
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
| `fig1_pace_seg_system.pdf` | How do shared training, static compilation, visual risk, measured cost, and runtime selection form one system? | `paper/figures/archify/fig1_pace_seg_system/candidate.json` |
| `fig2_elastic_deployment.pdf` | How are four related capacities extracted and deployed as static engines? | `paper/figures/archify/fig2_elastic_deployment/candidate.json` |
| `fig3_dataset_overview.png` | What clean and adverse visual conditions are evaluated? | `paper/figures/scripts/render_qualitative_examples.py` |
| `fig4_flops_vs_latency.pdf` | Why is proportional FLOPs insufficient for a latency budget? | `paper/figures/scripts/plot_flops_latency.py` |
| `fig5_rq2_near_parity.pdf` | Does shared elastic training preserve matched-capacity segmentation quality? | `paper/figures/scripts/plot_rq2_near_parity.py` |
| `fig6_routing_decision_example.pdf` | How can one image and risk vector yield different feasible decisions under the same budget? | `paper/figures/scripts/plot_routing_decision_example.py` |
| `fig7_router_deployment_matched.pdf` | Does candidate-specific hard-budget routing improve quality under measured complete route costs? | `paper/figures/scripts/plot_router_deployment_matched.py` |
| `fig8_qualitative_grid.png` | How do the four capacities and the selected output differ on held-out images? | `paper/figures/scripts/render_qualitative_examples.py` |
| `figS1_hardest_failure.png` | What does the predetermined highest-error ACDC case look like? | `paper/figures/scripts/render_qualitative_examples.py` |
| `figS2_route_cost_expansion.pdf` | How much larger is the complete routing path than isolated candidate inference? | `paper/figures/scripts/plot_route_cost_expansion.py` |

The raster panels are deterministic assemblies of real dataset inputs, ground truth,
and Run A predictions. Their IDs, hashes, decisions, and exact confusion-matrix audit
are recorded in `paper/figures/qualitative/generated/render_manifest.json`.

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
- Parent manuscript snapshot tag: `ivc-qualitative-v6-20260930`
- Previous visual-scientific release: `ivc-visual-scientific-v3-20260929`
- Revision v7 source: this folder, `paper/figures/`, and `paper/tables/`
