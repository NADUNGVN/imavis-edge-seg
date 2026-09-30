# Qualitative figure evidence gate and renderer

The repository now freezes the qualitative selection in
`selection_manifest.json`. The selected IDs are derived from canonical Run A
held-out confusion matrices and the versioned dataset manifests; selection therefore
does not require access to the licensed RGB files. Final rendering still requires
the Cityscapes/ACDC trees and the recorded Run A checkpoint, neither of which is
committed to this repository.

The locked primary selection protocol is:

1. use the router held-out half only (odd indices of each evaluation loader);
2. for each ACDC condition (`fog`, `night`, `rain`, `snow`), compute per-image
   large-candidate error with the same ignore-label handling as evaluation;
3. select the image closest to the condition median error, breaking ties by the
   lexicographic image identifier;
4. render `Input | Ground truth | Tiny | Small | Medium | Large | PACE-Seg | Error`
   with the official 19-class Cityscapes palette;
5. use policy D's deployment-matched E3 operating point at the medium route-cost
   budget for the `PACE-Seg` column;
6. verify every source checksum and require regenerated confusion matrices to match
   the canonical dump exactly before writing a figure;
7. retain the highest-error ACDC held-out image as a separately labelled failure
   example, not as part of the median-case main grid.

This median-error rule prevents selecting only visually successful examples. The
routed column is now methodologically unlocked by the deployment-matched Run A--C
rerun; only the uncommitted licensed raster/checkpoint assets prevent local rendering.

Regenerate the selection manifest anywhere:

```bash
python paper/figures/scripts/select_qualitative_examples.py
```

Render the figures on the training server:

```bash
python paper/figures/scripts/render_qualitative_examples.py \
  --checkpoint outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt \
  --output-dir paper/figures/qualitative/generated
```

The renderer produces a five-condition dataset overview, the main qualitative grid,
a separate hardest-case panel, and a render manifest containing decisions and output
hashes. Required inputs are intentionally not replaced by synthetic or manually
edited imagery.
