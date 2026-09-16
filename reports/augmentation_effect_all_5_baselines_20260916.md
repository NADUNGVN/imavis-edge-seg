# Augmentation effect across all 5 baseline architectures — 2026-09-16

Completes the last 3 of 5 required baselines' augmented runs (`bisenetv2`,
`ddrnet23_slim`, `mobilenetv3_deeplabv3`, all seed0, 100k steps, identical recipe
to their existing no-augmentation runs). **Corrects**
`reports/segformer_augmentation_investigation_20260915.md`'s hypothesis, which was
built from only 2 data points (`fast_scnn`, `segformer_b0`) and does not survive
contact with the other 3.

## Full picture (Cityscapes delta = aug - no-aug; ACDC avg = mean over fog/night/rain/snow deltas)

| model | params (M) | no-aug Cityscapes mIoU | Cityscapes delta | ACDC avg delta |
|---|---:|---:|---:|---:|
| `fast_scnn` | 1.136 | 0.4080 | **+0.1147** | **+0.1071** |
| `bisenetv2` | 1.706 | 0.5811 | -0.0043 | +0.0033 |
| `segformer_b0` | 3.719 | 0.5665 | -0.0152 | +0.0155 |
| `ddrnet23_slim` | 5.220 | 0.6266 | +0.0003 | -0.0045 |
| `mobilenetv3_deeplabv3` | 11.025 | 0.5773 | **+0.0259** | **+0.0447** |

## The 2026-09-15 hypothesis is wrong

The prior investigation (2 data points: smallest model gains huge, a mid-size
model trades off slightly) proposed a monotonic "smaller/more-underfit models gain
more from augmentation" story. **This predicts `mobilenetv3_deeplabv3` — the
*largest* model here, 2.1x `ddrnet23_slim`'s params and ~3x `segformer_b0`'s —
should show the *smallest* gain or a trade-off like `segformer_b0`/`ddrnet23_slim`.
It does not**: it shows the second-largest blanket gain, on both Cityscapes and
every ACDC condition, behind only `fast_scnn`.

## A better-supported explanation: gains are large at *both* ends of a
"how far from a comfortable fit" axis, not a monotonic function of size

- **`fast_scnn` (smallest, 1.1M)**: catastrophically underfit without
  augmentation (Cityscapes no-aug mIoU 0.408, far the worst of all 5) —
  augmentation acts as an effective-data multiplier a too-small model badly
  needs. Large gain, mechanism: **underfitting relief**.
- **`mobilenetv3_deeplabv3` (largest, 11.0M)**: despite having 2.1x
  `ddrnet23_slim`'s parameters, its no-aug Cityscapes mIoU (0.5773) is *worse*
  than `ddrnet23_slim`'s (0.6266) — consistent with overfitting the small
  (4575-image) training set at that budget rather than being well-fit.
  Augmentation acts as regularization a too-large, likely-overfitting model
  needs. Large gain, mechanism: **overfitting relief** — the opposite mechanism
  from `fast_scnn`, same net direction of effect.
- **`bisenetv2` (1.7M), `segformer_b0` (3.7M), `ddrnet23_slim` (5.2M)**: all
  land in a comfortable middle -- reasonable no-aug Cityscapes mIoU (0.567-0.627,
  the 3 best no-aug numbers of the 5) without being so large they overfit.
  Augmentation's effect here is small and mixed: near-zero to small-negative on
  Cityscapes (-0.0152 to +0.0003), small and inconsistent-signed on ACDC average
  (-0.0045 to +0.0155) -- neither a strong regularization need nor a strong
  underfitting-relief need, so augmentation mostly just trades a fixed
  training-step budget between the clean and augmented distributions with no
  large net win either way.

This is a **bimodal / U-shaped relationship with model size relative to the
dataset+budget**, not the earlier monotonic "smaller gains more" story. Params
count is a proxy, not the real mechanism -- what actually matters is whether a
given architecture (its parameter count *and* its inductive biases -- BiSeNetV2's
dual-branch design or DDRNet's design evidently fit this data comfortably at
sizes similar to or larger than `fast_scnn`/`segformer_b0`) lands under-, well-,
or over-fit at this specific (4575-image, 100k-step) budget.

## What this means for the manuscript

**Do not claim "augmentation helps proportionally to how small/simple the
model is."** The honest claim: augmentation's benefit is architecture- and
budget-dependent, largest for models that are (for whatever reason -- too little
capacity, or too much capacity relative to the training set) not comfortably
fit by the no-augmentation baseline, and small/mixed for models that already are.
Whether a given architecture falls in the "needs augmentation" or "already
comfortable" bucket is not predictable from parameter count alone without
knowing its no-augmentation fit quality first.

## Not yet done

- No direct evidence for the "overfitting" mechanism beyond the indirect signal
  (worse no-aug val mIoU despite more params) -- a train-set accuracy/loss
  comparison would confirm the overfitting story more directly for
  `mobilenetv3_deeplabv3` specifically.
- Only seed0 for each of these 3 newly-augmented runs -- no seed-repeat check
  yet (unlike `fast_scnn`'s 3-seed go/no-go work).
- All 5 baseline required slots (`RESEARCH_PLAN.md` §7) now have both
  augmented and non-augmented runs at seed0. `pidnet_s` remains unneeded
  (`ddrnet23_slim` satisfies "PIDNet-S or DDRNet-23-slim"); `hard`/`ucpnet`
  remain `NotImplementedError` (no public release found).
