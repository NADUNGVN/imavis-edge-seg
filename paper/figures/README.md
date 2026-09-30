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
| `archify/fig2_candidate_family_v11/` | Editable V11 candidate-family source plus Archify validation receipts |
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
| `scripts/plot_candidate_family_summary.py` | Generates the V10 candidate-family summary directly from `paper/tables/candidate_family.csv` |
| `scripts/plot_routing_decision_example_v10.py` | Generates the enlarged V10 hardware-conditioned decision example |
| `scripts/assemble_visual_story_v10.py` | Reframes audited composites as enlarged V10 Introduction, Method, and supplementary panels |
| `scripts/plot_candidate_family_silhouettes_v11.py` | Generates the V11 candidate-family silhouette and metric strip from `candidate_family.csv` |
| `scripts/plot_visual_routing_pipeline_v11.py` | Builds the V11 raw-image-to-routed-output method pipeline from audited artifacts |
| `scripts/plot_complete_routing_decision_v11.py` | Builds one complete policy-D decision with risks, route costs, budget, and error overlay |
| `scripts/plot_flops_latency_v11.py` | Generates the enlarged V11 RQ1 hardware-cost plot |
| `scripts/plot_rq2_near_parity_v11.py` | Generates the enlarged V11 RQ2 near-parity plot |
| `scripts/plot_routing_decision_example_v11.py` | Generates the V11 same-image E3/E1 decision comparison |
| `scripts/plot_router_deployment_matched_v11.py` | Generates the enlarged V11 RQ3 complete-route frontier |
| `scripts/assemble_visual_story_v11.py` | Generates V11 Figures 1, 9, S1, S3, and S4 from the six frozen audited examples |
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
.venv\Scripts\python.exe paper\figures\scripts\plot_candidate_family_summary.py
.venv\Scripts\python.exe paper\figures\scripts\plot_routing_decision_example_v10.py
.venv\Scripts\python.exe paper\figures\scripts\assemble_visual_story_v10.py
.venv\Scripts\python.exe paper\figures\scripts\plot_candidate_family_silhouettes_v11.py
.venv\Scripts\python.exe paper\figures\scripts\plot_visual_routing_pipeline_v11.py
.venv\Scripts\python.exe paper\figures\scripts\plot_complete_routing_decision_v11.py
.venv\Scripts\python.exe paper\figures\scripts\plot_flops_latency_v11.py
.venv\Scripts\python.exe paper\figures\scripts\plot_rq2_near_parity_v11.py
.venv\Scripts\python.exe paper\figures\scripts\plot_routing_decision_example_v11.py
.venv\Scripts\python.exe paper\figures\scripts\plot_router_deployment_matched_v11.py
.venv\Scripts\python.exe paper\figures\scripts\assemble_visual_story_v11.py
```

Archify remains the editable provenance source for architecture diagrams. The V11
publication candidate-family figure uses a source-backed Matplotlib silhouette because
it remains clearer at journal size; the Archify source and validation receipts are kept
beside it. Final diagrams and plots remain vector, while audited qualitative assemblies
remain publication-resolution PNG files. Figures S3 and S4 reuse the same six audited
cases as S1; S4 uses the fixed normalized crop $x=[0.25,0.75]$, $y=[0.36,1.00]$.
