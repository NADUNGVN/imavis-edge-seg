# PACE-Seg Overleaf package, scientific-story revision

This package is a self-contained `elsarticle` manuscript for submission preparation at *Image and Vision Computing*. The manuscript compiles from `main.tex`; all figures are editable vector graphics written in TikZ/PGFPlots under `figures/`.

## Upload to Overleaf

1. Download or locate `PACE_SEG_OVERLEAF_PACKAGE_V2_20260929.zip`.
2. In Overleaf, choose **New Project → Upload Project**.
3. Upload the ZIP file. Overleaf will unpack the complete folder structure.
4. Confirm that `main.tex` is selected as the main document.
5. Compile with pdfLaTeX. The project uses only packages available in a standard Overleaf TeX Live installation.

To update an existing project instead, upload `main.tex` and the entire `figures/` folder. Preserve the folder name because `main.tex` uses paths such as `\input{figures/fig1_system_overview.tex}`.

## Included explanatory figures

| Figure | Purpose | Canonical content |
|---|---|---|
| `fig1_system_overview.tex` | Explains the complete inference path | tiny probe, entropy, per-condition candidate calibrators, measured route costs, hard-budget policy D |
| `fig2_protocol_split.tex` | Makes leakage boundaries understandable | even-index fit half, locked policy, odd-index held-out half, ignore-mask mismatch |
| `fig3_flops_vs_latency.tex` | Visualizes the RQ1 cost mismatch | 110.2× FLOPs versus 10.3–19.5× measured latency |
| `fig4_route_overhead.tex` | Compares real backend overhead | E3 and E1 complete warm-route medians |
| `fig5_router_result.tex` | Shows the conditional routing diagnostic | 51/16/2 on 69 fair cells and D 0/120 versus A 51/120 violations |
| `fig6_qat_heatmap.tex` | Summarizes the bounded QAT evidence | worst-condition degradation for three runs and four levels |

The figures intentionally avoid synthetic street scenes or generated segmentation masks. Such imagery could be mistaken for experimental evidence. If a qualitative image panel is added later, it should use actual licensed Cityscapes/ACDC samples and predictions from the canonical checkpoints.

## Recommended qualitative panel to add from real outputs

One additional real-image figure would materially improve accessibility. Use two examples: one Cityscapes image and one difficult ACDC/night or ACDC/rain image. For each row, show:

1. RGB input;
2. ground truth;
3. tiny prediction;
4. policy-D selected prediction;
5. large prediction;
6. a short annotation containing probe entropy, selected level, budget, and measured route cost.

Select examples by a rule fixed before visual inspection, such as median held-out probe risk and 90th-percentile held-out probe risk. Do not select only visually favorable examples. State the selection rule in the caption and respect dataset redistribution licenses.

## Evidence boundaries retained in the manuscript

- Calibration is fitted separately for Cityscapes and each ACDC condition.
- The deployment condition is assumed known or configured.
- E1 and E3 share segmentation predictions; they are not independent accuracy replications.
- The 120 operating cells are correlated.
- The 69 fair cells are conditionally defined by policy A's feasibility.
- D's zero budget violations are partly a consequence of its hard constraint.
- Hardware accuracy is an offline replay with directly measured complete warm-route costs.
- QAT uses fake quantization and does not establish compiled INT8 accuracy.
- No energy result is claimed.
- Fit-time entropy excludes ignore pixels while deployment entropy does not; the routing comparison is therefore diagnostic rather than a frozen headline.

## Canonical repository snapshot

- Manuscript release tag: `ivc-scientific-story-v2-20260929`
- Frozen evidence snapshot: `ac65b562ed1cc33b7a30a63b2451924b156f8bdd`
- Evidence baseline: `1c4d42a7ea525931154ebb2f9015d898ef7771a5`

The stale seed0 E3 filename and immutable-tag documentation issues identified in the first review package have been corrected. The remaining submission gate is a deployment-matched router-calibration rerun or an editorial decision to retain RQ3 explicitly as diagnostic evidence.
