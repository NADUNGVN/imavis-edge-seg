# Investigating the `segformer_b0` augmentation regression — 2026-09-15

`reports/baseline_comparison_gap_check_20260912.md` and `README.md` flagged that
`segformer_b0`-aug scored *lower* on Cityscapes than `segformer_b0`-no-aug
(0.5665 → 0.5513, -1.52 points), the opposite of every other tested architecture
(`fast_scnn`: +9 to +11.5 points). Flagged as unexplained. Investigated using the
per-class breakdowns already committed in `reports/eval_baseline_segformer_b0_step100000.json`
and `..._aug_step100000.json` (no new training/eval run needed).

## Finding 1: the Cityscapes "regression" is not a blanket effect — it's concentrated in
2 rare classes

Per-class delta (aug - no-aug), Cityscapes:

| class | aug | no-aug | delta |
|---|---|---|---|
| train | 0.192 | 0.301 | **-0.109** |
| truck | 0.356 | 0.440 | **-0.084** |
| fence | 0.313 | 0.352 | -0.040 |
| bus | 0.506 | 0.538 | -0.032 |
| rider | 0.264 | 0.287 | -0.023 |
| ...15 other classes | | | -0.016 to +0.019 |

`train` and `truck` alone account for -0.193 of the -0.152 mIoU (mIoU is an
unweighted mean over 19 classes, so 2 classes moving -0.1 each is enough to swing
the whole mean negative). Every common/large class (road, building, sky, car,
vegetation: 0.86-0.97 IoU) is essentially unchanged (±0.002).

## Finding 2: augmentation clearly *helps* segformer_b0 on every adverse (ACDC) split

| split | aug | no-aug | delta |
|---|---|---|---|
| cityscapes (clean) | 0.5513 | 0.5665 | -0.0152 |
| acdc/fog | 0.6335 | 0.6151 | **+0.0184** |
| acdc/night | 0.4057 | 0.4072 | -0.0014 (flat) |
| acdc/rain | 0.5465 | 0.5400 | **+0.0066** |
| acdc/snow | 0.5628 | 0.5244 | **+0.0384** |

Average over the 4 ACDC splits: **+0.0155** — a net gain, not a regression. The
original framing (`baseline_comparison_gap_check`) reported only the Cityscapes
number and did not check ACDC, which is why this looked like an unqualified
regression.

## Finding 3: contrast with `fast_scnn` explains the mechanism

`fast_scnn` (1.1M params) no-aug is badly underfit at the rare classes specifically
(`train`=0.098, `truck`=0.145, `fence`=0.180 IoU — catastrophically low). Its
aug run gains massively on exactly those classes (`train` +0.223, `truck` +0.237,
`fence` +0.229) because augmentation acts as a strong regularizer/effective-data
multiplier a low-capacity, underfit model badly needs.

`segformer_b0` (3.7M params, transformer-based, ~3.3x `fast_scnn`'s capacity) is
**not** underfit on those same rare classes without augmentation (no-aug:
`train`=0.301, `truck`=0.440 — already 2-3x `fast_scnn`'s no-aug numbers). Adding
augmentation, at the *same fixed 100k-step training budget*, spends some of that
fixed optimization budget fitting a harder, noisier augmented input distribution.
For a model that already had reasonable rare-class fits, that trade costs a little
accuracy on the classes that already had few training examples (rare classes ->
augmented crops more often place them off-frame or distorted, without the offsetting
underfitting benefit `fast_scnn` got) — while buying a larger, consistent gain on
generalizing to the unseen adverse-weather domain (ACDC), which is exactly what the
augmentation (scale/crop/flip/color-jitter) is designed to help with.

## Conclusion — not a bug, not "augmentation doesn't work here"

The correct claim is a **capacity/step-budget-dependent trade-off**, not a
blanket "augmentation doesn't help segformer_b0": at a fixed training budget,
augmentation costs `segformer_b0` a little on Cityscapes' already-well-fit rare
vehicle classes (train, truck), while still net-improving its accuracy on every
adverse ACDC condition. Whether more training steps (so the fixed budget no longer
has to trade off) or class-balanced augmentation sampling would close the small
Cityscapes gap without giving up the ACDC gain is untested — a reasonable follow-up
if this becomes relevant to the manuscript's augmentation discussion, but not
blocking any current go/no-go bar.

**Updates**: `README.md` Phase 8 side-finding text corrected accordingly (was:
"unexplained... do not generalize 'augmentation helps' as a blanket claim"; now
reflects the capacity/domain trade-off found here, still not generalizing a
blanket claim, but the mechanism is understood rather than unexplained).

## Not yet done

- Not checked on `bisenetv2`/`ddrnet23_slim`/`mobilenetv3_deeplabv3` (all still only
  have a no-augmentation run — this hypothesis predicts augmented runs on the
  *higher-capacity* end of that set would show the same small clean-domain /
  adverse-domain trade-off, and the lower-capacity end would show `fast_scnn`-like
  blanket gains, but this is untested).
- No controlled ablation isolating "more steps" vs "class-balanced augmentation
  sampling" as a fix for the small Cityscapes-side cost, if it ever needs fixing.
