# PACE-Seg Overleaf package, visual-scientific revision v3

This folder is a self-contained `elsarticle` manuscript for *Image and Vision
Computing*, Special Issue: Complex Environment Vision. Compile `main.tex` with
pdfLaTeX. The four included figures are vector PDFs and require no shell escape.

## Upload to Overleaf

1. Upload `PACE_SEG_OVERLEAF_PACKAGE_V3_20260929.zip` as a new project, or upload
   `main.tex` plus the complete `figures/` folder to the existing project.
2. Select `main.tex` as the main document.
3. Compile with pdfLaTeX.

The repository also provides `paper/scripts/build_manuscript_pdf.ps1`. Running it
from the repository updates the stable review artifact at
`output/pdf/PACE-Seg_IVC_Manuscript.pdf` from this package's `main.tex`.

## Main-paper visual narrative

| Figure | Scientific question | Reproducible source |
|---|---|---|
| `fig1_pace_seg_system.pdf` | How do shared training, static compilation, visual risk, measured cost, and runtime selection form one system? | `paper/figures/archify/fig1_pace_seg_system/candidate.json` |
| `fig2_elastic_deployment.pdf` | How are four related capacities extracted and deployed as static engines? | `paper/figures/archify/fig2_elastic_deployment/candidate.json` |
| `fig3_flops_vs_latency.pdf` | Why is proportional FLOPs insufficient for a latency budget? | `paper/figures/scripts/plot_flops_latency.py` |
| `fig4_rq2_near_parity.pdf` | Does shared elastic training preserve matched-capacity segmentation quality? | `paper/figures/scripts/plot_rq2_near_parity.py` |

The diagrams retain editable Archify JSON and exported SVG in the repository. The
data plots parse canonical JSON/CSV artifacts and do not embed experimental numbers
manually.

## Deliberately gated figures

- **Qualitative vision panel:** not included because the repository snapshot does not
  contain licensed RGB images, ground-truth masks, and checkpoint predictions. The
  deterministic median-error selection rule is documented in
  `paper/figures/qualitative/README.md`.
- **Router quality--latency plot:** not manuscript-final because calibration fitting
  uses a ground-truth ignore mask while deployment cannot. The correction and rerun
  gate is documented in `paper/figures/ROUTER_FIGURE_GATE.md`.

No generated or manually retouched image is used as experimental evidence.

## Evidence boundaries retained in the manuscript

- E1 and E3 reuse segmentation predictions and are not independent accuracy replications.
- Operating cells are correlated; 69 fair cells are conditionally defined by A feasibility.
- D's zero violations are partly structural under the hard budget.
- Hardware accuracy is an offline replay using measured complete warm-route costs.
- QAT uses fake quantization and does not establish compiled INT8 accuracy.
- No energy result is claimed.
- RQ3 remains conditional until the probe feature is identical at fitting and deployment.

## Repository references

- Frozen evidence snapshot: `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`
- Previous scientific-story release: `ivc-scientific-story-v2-20260929`
- Visual-scientific v3 source: this folder and `paper/figures/`

The final v3 release tag should be created only after PDF compilation and page-by-page
inspection.
