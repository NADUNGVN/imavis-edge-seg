# PACE-Seg manuscript

This directory is the LaTeX working copy of the living manuscript skeleton in
[`docs/MANUSCRIPT_SKELETON.md`](../docs/MANUSCRIPT_SKELETON.md).

## Build

On Overleaf, set `main.tex` as the main document and use the default LaTeX build
configuration. Locally, a standard IEEEtran installation can be built with:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This is a working skeleton, not a submission-ready manuscript. Every `TODO`
marker must be resolved from a landed, reproducible experiment before making a
strong claim.

## Structure

- `main.tex`: IEEE manuscript entry point.
- `sections/`: one file per manuscript section.
- `references.bib`: bibliography to be expanded after the systematic literature review.
- `figures/` and `tables/`: reserved for generated artifacts (currently empty).
