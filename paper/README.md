# PACE-Seg paper workspace

This directory separates the current submission package, immutable review rounds, local release archives, and earlier manuscript drafts.

## Canonical locations

| Path | Role | Status |
|---|---|---|
| [`submission/ivc_2026-09-30_v11_final-figure-map/`](submission/ivc_2026-09-30_v11_final-figure-map/) | Final nine-figure visual narrative with S1--S4 audited supplementary views | Canonical writing and visual package |
| [`submission/ivc_2026-09-30_v10_research-visuals/`](submission/ivc_2026-09-30_v10_research-visuals/) | Shorter research-oriented manuscript with enlarged evidence panels and a quantitative candidate-family summary | Preserved review input |
| [`submission/ivc_2026-09-30_v9_figure-revision/`](submission/ivc_2026-09-30_v9_figure-revision/) | Manuscript with the revised input--ground-truth--route narrative, candidate-family diagram, and expanded error gallery | Preserved review input |
| [`submission/ivc_2026-09-30_v8_early-visuals/`](submission/ivc_2026-09-30_v8_early-visuals/) | Manuscript with early data onboarding, literature positioning table, and audited input-to-output routing pipeline | Preserved review input |
| [`submission/ivc_2026-09-30_v7_expanded/`](submission/ivc_2026-09-30_v7_expanded/) | Manuscript with a real-image routing explainer, route-cost expansion, and condition-level router table | Preserved review input |
| [`submission/ivc_2026-09-30_v6_qualitative/`](submission/ivc_2026-09-30_v6_qualitative/) | Manuscript with audited real-data overview, qualitative comparison, and predetermined failure panel | Preserved review input |
| [`submission/ivc_2026-09-30_v5_visual-data/`](submission/ivc_2026-09-30_v5_visual-data/) | Deployment-matched manuscript with onboarding tables and the qualitative rendering pipeline | Preserved review input |
| [`submission/ivc_2026-09-30_v4/`](submission/ivc_2026-09-30_v4/) | Deployment-matched Elsevier manuscript before the visual-data addendum | Preserved review input |
| [`submission/ivc_2026-09-29_v3/`](submission/ivc_2026-09-29_v3/) | Visual-scientific package before the deployment-matched router rerun | Preserved review input |
| [`submission/ivc_2026-09-29_v2/`](submission/ivc_2026-09-29_v2/) | Scientific-story revision before the visual rebuild | Preserved review input |
| [`submission/ivc_2026-09-29/`](submission/ivc_2026-09-29/) | Prior reviewed package before the scientific-story revision | Preserved review input |
| [`reviews/2026-09-30_deployment-matched-final/`](reviews/2026-09-30_deployment-matched-final/) | Reviewer #2 audit of final RQ3 evidence and v4 manuscript | Current review record |
| [`reviews/2026-09-29_scientific-story-audit/`](reviews/2026-09-29_scientific-story-audit/) | Claim-status audit for the RQ-organized manuscript | Preserved review record |
| [`reviews/2026-09-29_gpt-strict/`](reviews/2026-09-29_gpt-strict/) | Markdown manuscript, two Reviewer \#2 audits, and readiness statement | Completed review round |
| [`reviews/README.md`](reviews/README.md) | Naming and evidence protocol for future review rounds | Active protocol |
| `releases/` | Locally generated upload archives | Intentionally ignored by Git |

The root-level `main.tex`, `sections/`, `references.bib`, and `tables/` are the earlier
modular manuscript skeleton. They are retained for provenance but are not the current
submission source. `figures/` is now the canonical reproducible figure workspace used
by v11. The earlier `../manuscript/latex/main.tex` is also non-canonical.

## Build the current manuscript

Open [`submission/ivc_2026-09-30_v11_final-figure-map/main.tex`](submission/ivc_2026-09-30_v11_final-figure-map/main.tex)
with pdfLaTeX. On Overleaf, upload the entire `ivc_2026-09-30_v11_final-figure-map` directory or the
v11 archive from `output/overleaf/`, preserve the `figures/` and `tables/` subdirectories, and select
`main.tex` as the main document.

For a reproducible local build that also updates the stable review PDF, run:

```powershell
powershell -ExecutionPolicy Bypass -File paper\scripts\build_manuscript_pdf.ps1
```

The command compiles the canonical `main.tex` and replaces
`output/pdf/PACE-Seg_IVC_Manuscript.pdf`. This stable PDF is the file to distribute
for reviews. Whenever the canonical TeX or its figures change, rebuild and commit the
updated PDF in the same change.

## Evidence discipline

- Numerical claims must trace to reports or raw artifacts listed in the submission evidence manifest.
- A review round must record its input snapshot, reviewer report, mandatory revisions, and final readiness decision.
- A revised submission receives a new dated directory; previous review inputs are never overwritten.
- Generated PDFs and ZIP archives are release products and are not committed.
