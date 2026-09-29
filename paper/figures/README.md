# PACE-Seg figure workspace

This directory contains editable sources, reproducible generators, and final journal
vectors. A figure is included in the main manuscript only when it answers a central
scientific question and traces to repository evidence.

## Layout

| Path | Role |
|---|---|
| `archify/fig1_pace_seg_system/` | Editable, source-backed system overview |
| `archify/fig2_elastic_deployment/` | Editable elasticity/deployment diagram |
| `archify/VALIDATION.md` | Archify version, hashes, gate results, and visual-review outcome |
| `scripts/export_archify_svg.py` | Calls Archify's vector exporter through Chromium |
| `scripts/svg_to_vector_pdf.py` | Converts exported SVG to tightly cropped vector PDF |
| `scripts/plot_flops_latency.py` | Generates RQ1 Figure 3 from canonical FLOPs/latency artifacts |
| `scripts/plot_rq2_near_parity.py` | Generates RQ2 Figure 4 from six evaluation JSON files |
| `generated/` | Final SVG/PDF and high-resolution plot previews |
| `qualitative/` | Evidence gate and deterministic selection protocol for a future real-image panel |
| `ROUTER_FIGURE_GATE.md` | Required correction/rerun before the router quality--latency plot is frozen |

## Rebuild

From the repository root, using the project virtual environment:

```powershell
.venv\Scripts\python.exe paper\figures\scripts\export_archify_svg.py `
  paper\figures\archify\fig1_pace_seg_system\fig1_pace_seg_system.html `
  paper\figures\generated\fig1_pace_seg_system.svg
.venv\Scripts\python.exe paper\figures\scripts\svg_to_vector_pdf.py `
  paper\figures\generated\fig1_pace_seg_system.svg `
  paper\figures\generated\fig1_pace_seg_system.pdf
.venv\Scripts\python.exe paper\figures\scripts\plot_flops_latency.py
.venv\Scripts\python.exe paper\figures\scripts\plot_rq2_near_parity.py
```

Repeat the first two commands for Figure 2. Final diagrams and plots remain vector;
PNG files are previews only.
