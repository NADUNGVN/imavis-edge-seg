# PACE-Seg paper workspace

This directory separates the current submission package, immutable review rounds, local release archives, and earlier manuscript drafts.

## Canonical locations

| Path | Role | Status |
|---|---|---|
| [`submission/ivc_2026-09-29/`](submission/ivc_2026-09-29/) | Current self-contained Elsevier `elsarticle` manuscript and six TikZ figures | Canonical writing package |
| [`reviews/2026-09-29_gpt-strict/`](reviews/2026-09-29_gpt-strict/) | Markdown manuscript, two Reviewer \#2 audits, and readiness statement | Completed review round |
| [`reviews/README.md`](reviews/README.md) | Naming and evidence protocol for future review rounds | Active protocol |
| `releases/` | Locally generated upload archives | Intentionally ignored by Git |

The root-level `main.tex`, `sections/`, `references.bib`, `figures/`, and `tables/` are the earlier modular manuscript skeleton. They are retained for provenance but are not the current submission source. The earlier `../manuscript/latex/main.tex` is also non-canonical.

## Build the current manuscript

Open [`submission/ivc_2026-09-29/main.tex`](submission/ivc_2026-09-29/main.tex) with pdfLaTeX. On Overleaf, upload the entire `ivc_2026-09-29` directory or the local archive from `releases/`, preserve the `figures/` subdirectory, and select `main.tex` as the main document.

## Evidence discipline

- Numerical claims must trace to reports or raw artifacts listed in the submission evidence manifest.
- A review round must record its input snapshot, reviewer report, mandatory revisions, and final readiness decision.
- A revised submission receives a new dated directory; previous review inputs are never overwritten.
- Generated PDFs and ZIP archives are release products and are not committed.
