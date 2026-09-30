# PACE-Seg Figure Toolchain (V13)

V13 uses one paper-wide visual contract and keeps every figure reproducible from
repository sources. External figure skills guide composition and QA; canonical
experiment artifacts and the active model implementation remain the evidence
sources.

## Skill revisions

| Tool / skill | Repository commit | Role in V13 |
|---|---|---|
| Icarus-Figures | `0bcaa40c07ef0dc6ea1506ab147ce0a4f5a542fd` | Composition and cross-figure QA |
| ml-architecture-diagram-skill | `94b074e8f19de8b1be199343470fbd9a2b1bb3b4` | Figure 2 Architecture IR and semantic audit |
| publication-chart-skill | `4298fee11554de95038a4d9c303929eb5a5276fb` | Figures 5, 6, 8 and table QA |
| tikz-scientific-figures | `0f0918c84db0b715bed476b1f607cd6e72cd35d4` | Conceptual and routing schematics |
| tikz-academic | `720e801f2af0b6731ca0aed35c44565c481ebcab` | Compact diagram patterns and fallback layouts |

## Executable environment

- Python 3.12 through the repository `uv` environment.
- Matplotlib 3.11.2, NumPy 2.5.3, pandas 3.0.6, Pillow 12.3.0.
- `pubfig` 0.3.0, `pubtab` 1.0.2, `paperfig` 0.6.0.
- `ml-architecture-diagram` 0.1.0.
- Tectonic 0.17.0 for manuscript compilation.
- Poppler 25.07.0 for PDF rendering and visual inspection.

## Reproducibility rules

1. Numerical figures parse canonical JSON/CSV artifacts; displayed values are not
   copied into plotting code when a machine-readable source exists.
2. Figure 2 follows `PaceSegSupernet.forward`, static extraction in `subnet.py`,
   and the active elasticity configuration in `config.py`.
3. Qualitative figures reuse the audited prediction composites and frozen image
   identifiers recorded by the existing manifests.
4. Redesigned plots and schematics export PDF, SVG, and PNG from one editable
   source.
5. The V12 submission directory is immutable. V13 sources live under
   `paper/figures/v13/` and `paper/submission/ivc_2026-09-30_v13_figure-system/`.

## Validation note

`ml-arch lint-publication` flags the generic publication IR as extremely wide and
reports four automatic-route crossings. That automatic layout is retained only as
an auditable intermediate. The manuscript uses `render_fig2_architecture_v13.py`,
which preserves the reviewed IR while reflowing the architecture, candidate cards,
and backend fan-out into the final two-panel journal composition.
