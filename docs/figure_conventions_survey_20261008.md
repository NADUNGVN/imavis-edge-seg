# Figure conventions survey: segmentation papers (2026-10-08)

Scope: an efficient, dynamic-inference segmentation paper (Cityscapes + ACDC fog/night/rain/snow, per-image routing to T/S/M/L on edge accelerators), targeting IVC (Elsevier, 5p two-column elsarticle).

**Verification status.** I fetched and checked only the Elsevier artwork-sizing page (sizes, DPI and fonts in Section 5). The figure numbers and layout details for the papers below come from memory of the published versions. They are not re-checked against the PDFs. Before citing a figure number in the paper, open the arXiv HTML or PDF and confirm it.

---

## 1. Qualitative comparison grids

### (a) Observed conventions
- **SegFormer (NeurIPS 2021), Fig. 5 and appendix figs.** Rows are examples and columns are methods (SETR / DeepLabV3+ / SegFormer). Cityscapes crops are kept at the native 2:1 aspect. Differences are marked with **dashed or solid coloured boxes** and **zoomed insets** of thin or distant objects. There is no per-figure class legend. Masks use the pure Cityscapes palette, with no overlay on the image.
- **DAFormer (CVPR 2022), Fig. 5; HRDA (ECCV 2022), Fig. 4/5.** Columns: Image | baseline UDA | ours | GT. Each row is a different scene (about 4–6 rows). Rows include Cityscapes plus ACDC/DarkZurich in HRDA. **White boxes** mark the regions where the methods differ. The **GT ignore pixels are black** (void = 0,0,0). A class colour legend strip runs along the bottom across the full width.
- **Refign (WACV 2023) and the ACDC paper (Sakaridis, ICCV 2021), qualitative figs.** One row per condition (fog / night / rain / snow). Columns: input | several methods | GT. Ignore pixels are shown black, or white in some ACDC visualisations. Column headers sit at the top in small sans-serif text.
- **DDRNet (T-ITS 2022), PIDNet (CVPR 2023), BiSeNetV2 (IJCV 2021), STDC (CVPR 2021).** Real-time papers have few qualitative figures, usually a 3–4 row grid (Image | GT | competitors | ours), sometimes with **boxed regions**. PIDNet adds boundary/detail visualisations. Images are downscaled heavily and the masks are pure palette colour.
- **General patterns.**
  - Methods are columns and examples are rows: 3–6 rows, 4–6 columns.
  - GT is almost always shown.
  - Highlights are rectangles (often white or yellow, sometimes dashed). Error maps are rare in the main paper.
  - Overlays (alpha blending on the image) are uncommon for Cityscapes. Pure palette masks dominate.
  - Headers are column labels on top only, with no row labels unless the rows are conditions.

### (b) Recommended spec for our paper
- **Full width, 190 mm.** Use 4 rows for ACDC (one per condition: fog/night/rain/snow) plus an optional row for Cityscapes clean.
- **Columns (6):** Image | GT | Tiny | Large | Routed (ours) | Static baseline. Add a small badge in the corner of the Routed cell naming the chosen capacity, e.g. "S".
- **Cell size:** 190 mm / 6 is about 30.5 mm wide with 1 mm gaps. That gives cells of about 30 x 15 mm at 2:1, so the figure is about 80 mm tall for 5 rows.
- **Images:** centre-crop the native 1920x1080 ACDC frames to 2:1, or keep 16:9 for every column. Export raster at ≥300 dpi (combination art 500 dpi). Save the masks as PNG, not JPEG.
- **Colours:** use the pure Cityscapes 19-class palette. Draw ignore/void pixels **black** in GT and state this in the caption.
- **Highlights:** one dashed white box (0.5 pt) per row on the region that differs. An optional 2x zoom inset can go in the corner.
- **Legend:** one 19-class legend strip across the full width at the bottom, 6–7 pt text, two rows of swatches.
- **Headers:** 7 pt sans-serif column headers on top. Condition labels are rotated 90° on the left.

## 2. Dataset/condition overview figures

### (a) Observed conventions
- **ACDC (ICCV 2021), Fig. 1 teaser.** A grid with one column per condition (fog, night, rain, snow). Rows show the adverse image, its dense annotation, and the corresponding normal-condition image. There are condition headers on top and no axes.
- **Dark Zurich / Foggy Cityscapes papers.** These use the same "image over label" pairing.

### (b) Recommended spec
- **Full width, 190 mm, 2 or 3 rows by 4 columns.** Columns: Fog | Night | Rain | Snow, with an optional Clean (Cityscapes) column as a 5th.
- **Rows:** image, then GT. An optional third row shows the router's chosen capacity per image, or a bar of the capacity distribution per condition. That third row ties the overview to the routing method.
- **Size:** each cell is about 46 x 26 mm. Use 7 pt headers.

## 3. Per-image dynamic decision / routing / early-exit figures

### (a) Observed conventions
- **Dynamic Routing (Li et al., CVPR 2020), Fig. 1 and the route-visualisation fig.** A schematic of the scale-space path. Selected paths are highlighted with thick coloured edges, and unselected ones are grey or thin. Example images are placed next to the path diagrams to show that the routes depend on the input.
- **MESS (multi-exit segmentation, ECCV 2020) and Anytime Dense Prediction (Liu et al., CVPR 2021).** Show outputs at successive exits, left to right, with growing quality and FLOPs labels under each panel. Early exits are tied to "easy" images.
- **Han et al., Dynamic Neural Networks survey (TPAMI 2022).** Schematic figures that separate sample-wise, spatial and temporal dynamics. Uses a cascade of models with a gating/confidence decision diamond, and keeps the colour coding consistent per model size.

### (b) Recommended spec
- **Two figures.**
  - **Method schematic:** full width (190 mm) or 140 mm. Input, then a lightweight router (features and confidence), then four capacity blocks T/S/M/L. The selected block is solid coloured and the unselected blocks are grey outlines. Mark the accelerator/latency budget on the arrow.
  - **Example strip:** 4–6 test images sorted by router decision. Under each image show the predicted mask, the chosen capacity tag, its latency in ms, and its mIoU against the Large model.
- **Colour:** give each capacity one fixed colour across all figures. A sequential, colour-blind-safe scheme works, for example viridis stops T #440154, S #31688E, M #35B779, L #FDE725, or a blue ramp. Do not reuse Cityscapes palette colours.

## 4. Quality-vs-cost plots

### (a) Observed conventions
- **SegFormer (NeurIPS 2021), Fig. 1.** mIoU vs params (or FLOPs). Each model family is a line with markers, and our method is a bold red line with star or circle markers. Points carry text labels (B0…B5). Axes are linear.
- **DDRNet, PIDNet, STDC, BiSeNetV2 (Fig. 1 in each).** mIoU vs FPS (inference speed). Each method is a single scatter point with a text label, and ours is a red star. A dashed vertical line marks real time (30 FPS). The FPS axis is usually linear, sometimes log. There are no error bars.
- **Dynamic-inference papers (Anytime Dense Prediction; Han survey).** Accuracy vs FLOPs curves. The dynamic method is a continuous curve traced by sweeping a threshold, and static models are discrete points. The Pareto improvement is the main message. Log-scaled FLOPs are common when the range covers more than 10x.

### (b) Recommended spec
- **Single column, 90 mm wide by about 65–70 mm tall.** Make 2 subplots side by side (190 mm) if both Cityscapes and ACDC are shown.
- **x-axis:** measured latency (ms) on the target accelerator, with mean latency on linear axes. Use a log axis only if T to L spans more than 10x. FLOPs go in a second panel or a table.
- **y-axis:** mIoU (%).
- **Marks:**
  - Static T/S/M/L are filled circles in the capacity colours, joined by a thin grey dashed line (the static Pareto front).
  - The router operating curve is a solid line traced by the threshold sweep, with ours marked by a star.
  - The oracle router is a hollow marker.
  - Competitors (PIDNet, DDRNet, etc.) are grey markers with 6 pt labels.
- **Error bars:** ±1 std over 3 seeds, or latency p50 with the p90 marked. Always state n in the caption. Few related papers do this, so it would set our plot apart.
- **Annotations:** a shaded region or arrow labelled "better" toward the upper left, and a dashed budget line if there is a deadline.
- **Text:** 7 pt fonts throughout, 0.75–1 pt lines, 4–5 pt markers. Export as vector PDF/EPS.

## 5. Widths, palettes, Elsevier/IVC specifics

**Elsevier artwork sizing (verified from the Elsevier artwork sizing page):**

| Item | Requirement |
|---|---|
| Minimum width | 30 mm |
| Single column | 90 mm |
| 1.5 column | 140 mm |
| Full width | 190 mm |
| Halftone resolution | ≥300 dpi |
| Combination art | ≥500 dpi |
| Line art | ≥1000 dpi |
| Normal lettering | 7 pt printed |
| Sub/superscripts | ≥6 pt |

**Typical uses by width:**
- **90 mm:** plots and small schematics.
- **140 mm:** medium schematics.
- **190 mm:** qualitative grids, overview figure, method diagram.

**Practical settings:**
- Set matplotlib `figsize` to the final size in inches (90 mm = 3.54 in, 190 mm = 7.48 in) so fonts are not rescaled in LaTeX.
- Use a sans-serif font (Helvetica/Arial or DejaVu Sans) or match the Times text. Pick one and keep it consistent.
- Elsevier prints colour online at no cost. Check that each plot still reads in greyscale through marker shapes and line styles.
- **Cityscapes palette:** use the official 19 trainId colours, e.g. road (128,64,128), sidewalk (244,35,232), car (0,0,142), sky (70,130,180), vegetation (107,142,35), person (220,20,60). Render void as black.

## Summary of recommended specs

| Figure | Width | Content |
|---|---|---|
| Teaser/overview | 190 mm | 4 conditions (+ clean) x (image, GT, routed capacity) |
| Method schematic | 190 or 140 mm | Router plus T/S/M/L, selected path coloured |
| Qualitative grid | 190 mm | 4–5 rows x 6 cols (Img, GT, T, L, Static, Routed+badge), black void, dashed white boxes, bottom legend |
| Routing examples | 190 mm | Images sorted by chosen capacity with latency/mIoU tags |
| mIoU vs latency | 90 mm (or 2 x 90 mm) | Static Pareto dashed, router curve, star for ours, error bars |
