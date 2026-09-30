# PACE-Seg figure workspace

This directory contains editable sources, reproducible generators, and final journal
vectors. A figure is included in the main manuscript only when it answers a central
scientific question and traces to repository evidence.

## Layout

| Path | Role |
|---|---|
| `archify/fig1_pace_seg_system/` | Editable, source-backed system overview |
| `archify/fig2_elastic_deployment/` | Editable elasticity/deployment diagram |
| `archify/fig2_candidate_family_v9/` | V9 candidate-family and static-deployment diagram |
| `archify/VALIDATION.md` | Archify version, hashes, gate results, and visual-review outcome |
| `scripts/export_archify_svg.py` | Calls Archify's vector exporter through Chromium |
| `scripts/svg_to_vector_pdf.py` | Converts exported SVG to tightly cropped vector PDF |
| `scripts/plot_flops_latency.py` | Generates the RQ1 FLOPs/latency figure from canonical artifacts |
| `scripts/plot_rq2_near_parity.py` | Generates the RQ2 near-parity figure from six evaluation JSON files |
| `scripts/plot_router_deployment_matched.py` | Generates the RQ3 figure from six canonical E1/E3 replay JSON files |
| `scripts/plot_routing_decision_example.py` | Combines audited real-image panels, candidate risks, and E1/E3 measured costs into the routing explainer |
| `scripts/plot_input_output_pipeline.py` | Builds the audited raw-image--probe--risk--budget--output Method figure from canonical artifacts |
| `scripts/plot_route_cost_expansion.py` | Compares candidate-only p95 latency with complete warm-route medians |
| `scripts/assemble_visual_story_v9.py` | Reframes audited composites as the V9 Introduction, Method, and supplementary panels |
| `scripts/summarize_router_by_condition.py` | Generates the condition-level CSV and LaTeX table rows from six canonical replays |
| `scripts/select_qualitative_examples.py` | Freezes median-error and failure-case IDs without raster access |
| `scripts/render_qualitative_examples.py` | Audits hashes/predictions and renders the real-data panels on the server |
| `generated/` | Final SVG/PDF and high-resolution plot previews |
| `qualitative/` | Selection manifest, rendered panels, and complete output audit |
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
.venv\Scripts\python.exe paper\figures\scripts\plot_router_deployment_matched.py
.venv\Scripts\python.exe paper\figures\scripts\plot_routing_decision_example.py
.venv\Scripts\python.exe paper\figures\scripts\plot_input_output_pipeline.py
.venv\Scripts\python.exe paper\figures\scripts\plot_route_cost_expansion.py
.venv\Scripts\python.exe paper\figures\scripts\summarize_router_by_condition.py
.venv\Scripts\python.exe paper\figures\scripts\assemble_visual_story_v9.py
```

Repeat the Archify export commands for the V9 candidate-family source when it changes.
Final diagrams and plots remain vector; audited qualitative assemblies remain
publication-resolution PNG files.
