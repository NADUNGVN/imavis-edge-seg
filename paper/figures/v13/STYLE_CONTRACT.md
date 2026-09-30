# PACE-Seg V13 Visual Contract

## Scientific hierarchy

- Each figure defends one claim and has one visually dominant element.
- Architecture and routing figures read left to right.
- Numerical panels prioritize the measured relationship; annotations are limited
  to the conclusion needed to interpret the panel.
- Captions explain evidence and uncertainty; artwork uses short labels only.

## Typography and geometry

- Full-width target: 178 mm. Single-column target: 86 mm.
- Final-size body text: 8--9 pt; panel labels: 9 pt bold.
- Sans-serif figure text with Arial/Helvetica-compatible fallbacks.
- White background, dark-neutral ink, thin axes and connectors, no gradients,
  shadows, pseudo-3D blocks, or decorative icons.

## Candidate encoding

| Candidate | Color | Marker | Secondary encoding |
|---|---|---|---|
| Tiny | `#E69F00` | `o` | solid |
| Small | `#56B4E9` | `s` | dashed |
| Medium | `#009E73` | `D` | dash-dot |
| Large | `#0072B2` | `^` | dotted |

Candidate color is stable across architecture silhouettes, latency plots, and
qualitative badges. Color is always reinforced by a marker, label, border, or
line style.

## Policy encoding

| Policy | Color | Marker | Line |
|---|---|---|---|
| Policy A | `#6B7280` | `o` | dashed |
| Policy D configured | `#D55E00` | `s` | solid |
| Policy D pooled | `#009E73` | `^` | dash-dot |

Risk uses muted magenta (`#CC79A7`). Hardware cost uses slate (`#475569`). The
budget is a dark dashed rule (`#111827`). A selected route uses a thick border
and explicit label rather than color alone.

## Output contract

- Schematics and plots: editable source + PDF + SVG + PNG preview.
- Qualitative panels: deterministic assembly source + PNG at publication
  resolution + source manifest.
- All raster evidence remains unchanged except deterministic crop, scale, border,
  and label composition.
