# Qualitative figure evidence gate

Figure 5 must be assembled from real validation images, ground-truth masks, and
predictions produced by recorded PACE-Seg checkpoints. None of those raster assets
are present in this repository snapshot, so no qualitative panel is included in the
submission package yet.

The locked selection protocol is:

1. use the router held-out half only (odd indices of each evaluation loader);
2. for each ACDC condition (`fog`, `night`, `rain`, `snow`), compute per-image
   large-candidate error with the same ignore-label handling as evaluation;
3. select the image closest to the condition median error, breaking ties by the
   lexicographic image identifier;
4. render `Input | Ground truth | Tiny | Large` with the official 19-class
   Cityscapes palette;
5. record image ID, source paths, checkpoint IDs, and file hashes in a generated
   `selection_manifest.json`.

This median-error rule prevents selecting only visually successful examples. A routed
column remains gated by the fit/deployment risk-feature correction and router rerun.

Required inputs for the assembly script are intentionally not replaced by synthetic
or manually edited imagery.
