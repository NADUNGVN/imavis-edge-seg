# First real go/no-go signal: supernet vs. independently-trained baselines — 2026-09-12

`RESEARCH_PLAN.md` §11's Go criterion / RQ2's hypothesis asks whether subnets
extracted from the shared supernet land within 1.0-1.5 mIoU of independently-trained
models at the *same parameter budget* (no-go trigger: >2 mIoU loss). Three required
baselines finished their 100k-step from-scratch training runs and were evaluated
2026-09-12: `fast_scnn` (1.136M params), `bisenetv2` (1.706M params), `ddrnet23_slim`
(5.220M params) -- see `reports/first_full_supernet_run_100k_20260910.md` for the
supernet's own numbers (1.047M total params, shared across all 4 levels).

## The valid same-budget comparison: fast_scnn (1.136M) vs. supernet-large (~1.047M)

Both trained from scratch, 100k steps, no augmentation (landed after these runs
started), same data. This is the first apples-to-apples test of RQ2.

| dataset | fast_scnn | supernet-large (seed 0/1 avg) | gap (supernet − fast_scnn) |
|---|---:|---:|---:|
| cityscapes | 0.4080 | 0.4706 | **+0.0626** |
| acdc/fog | 0.4593 | 0.4854 | **+0.0261** |
| acdc/night | 0.2917 | 0.3290 | **+0.0373** |
| acdc/rain | 0.3891 | 0.4535 | **+0.0644** |
| acdc/snow | 0.3915 | 0.4530 | **+0.0615** |

**The supernet wins on all 5/5 splits**, by 2.6-6.4 mIoU points. This isn't just
clearing the no-go bar (>2 mIoU loss) -- it exceeds the Go hypothesis (subnets land
*within* 1.0-1.5 mIoU of independent training) in the *positive* direction, which
RQ2's hypothesis didn't even anticipate. Plausible explanation: the supernet's
in-place distillation (the "large" level is also the teacher every step, but the
sandwich-rule's smaller sampled levels get real KD signal) and boundary-aware loss may
be acting as effective regularization/multi-task signal that a single independently-
trained network at the same budget doesn't get. Not yet verified which mechanism
specifically -- an ablation (with/without distillation, RESEARCH_PLAN.md §10) would
tell.

## Also trained (informative, not valid same-budget comparisons)

| model | params | cityscapes mIoU | ratio vs. supernet's 1.047M |
|---|---:|---:|---:|
| mobilenetv3_deeplabv3 | 11.02M | 0.5773 | 10.5x (+ ImageNet-pretrained backbone) |
| bisenetv2 | 1.706M | 0.5811 | 1.6x |
| ddrnet23_slim | 5.220M | 0.6266 | 5.0x |

All three score higher than the supernet -- expected, since all three have more
capacity (and mobilenetv3 additionally has pretrained-backbone advantage). These are
useful as *upper-bound reference points* for what more capacity buys, not as go/no-go
evidence. Only `fast_scnn` is close enough in budget to be the valid test.

## SUPERSEDED 2026-09-12 (later same day) -- augmentation changes the picture

`fast_scnn` was re-trained with the newly-added `SegmentationTrainAugment` (random
scale+crop, flip, color jitter; `experiment_id=baseline_fast_scnn_seed0_aug`), same
100k steps, same everything else:

| dataset | fast_scnn (no aug) | fast_scnn (aug) | delta |
|---|---:|---:|---:|
| cityscapes | 0.4080 | **0.5227** | **+0.1147** |
| acdc/fog | 0.4593 | 0.5730 | +0.1137 |
| acdc/night | 0.2917 | 0.3813 | +0.0896 |
| acdc/rain | 0.3891 | 0.5004 | +0.1113 |
| acdc/snow | 0.3915 | 0.5055 | +0.1140 |

Augmentation is worth **~9-11.5 mIoU points** on this model/dataset size -- confirming
it as the highest-leverage change flagged in
`reports/first_full_supernet_run_100k_20260910.md`, and considerably larger than the
supernet's entire margin over fast_scnn above (2.6-6.4 points). **`fast_scnn` with
augmentation (0.5227 cityscapes) now beats the supernet without augmentation (0.4706)**
-- the "supernet wins 5/5" result above no longer holds as a fair comparison, since
only one side has augmentation now. It was correct and fairly measured *at the time*
(both sides lacked augmentation identically), but is superseded, not wrong.

**The real comparison now depends on `pace_seg_v1_seed2`** (SERVER-02, `seed=2`),
launched *after* `TrainingConfig.augment` defaulted to `True` -- it picked up
augmentation automatically, with no explicit override needed. Once it finishes, seed 2
vs. `fast_scnn_seed0_aug` is the first apples-to-apples (both augmented) test of the
go/no-go criterion. Until then, **no augmented same-budget comparison exists yet** --
treat the go/no-go question as open again, not resolved either way.

## RESOLVED (first data point) 2026-09-12 -- the real, augmented apples-to-apples test

`pace_seg_v1_seed2` (supernet, `seed=2`, trained *with* augmentation from the start)
finished and was evaluated against `fast_scnn_seed0_aug` -- both sides now augmented,
same 100k-step budget, same data:

| dataset | fast_scnn (aug) | supernet-large seed2 (aug) | gap (supernet − fast_scnn) |
|---|---:|---:|---:|
| cityscapes | 0.5227 | 0.5354 | +0.0127 |
| acdc/fog | 0.5730 | 0.5761 | +0.0031 |
| acdc/night | 0.3813 | 0.3870 | +0.0057 |
| acdc/rain | 0.5004 | 0.4883 | **−0.0121** |
| acdc/snow | 0.5055 | 0.5163 | +0.0108 |

Supernet wins 4/5 splits by small margins (0.3-1.3 mIoU points) and loses 1/5 (rain,
by 1.2 points). This is a **much more modest, and much more plausible, result than the
superseded no-augmentation comparison** -- close to parity, comfortably inside
RQ2's hypothesized "within 1.0-1.5 mIoU" band either direction, and nowhere near the
§11 no-go trigger (>2 mIoU loss). **Go/no-go: passes**, on this first augmented data
point.

This is one seed per side, not the 3 `RESEARCH_PLAN.md` §9 rule 8 wants for a
headline claim -- `pace_seg_v1_aug_seed0` and `fast_scnn_seed1_aug` are in progress
(SERVER-01/03) to start building that. With margins this small (~0.3-1.3 points,
smaller than the ~1-3 point seed-to-seed noise seen on the no-augmentation runs),
more seeds are needed before claiming a winner on any individual split with
confidence -- the *overall* "clears go/no-go" conclusion is robust already, but
"wins 4/5" specifically should not be over-read from a single seed pair.

## UPDATED 2026-09-13 -- 2 seeds per side: near-parity, not a supernet win

`pace_seg_v1_aug_seed0` finished, giving 2 augmented supernet seeds (seed0, seed2);
`fast_scnn_seed1_aug` also finished, giving 2 augmented fast_scnn seeds (seed0,
seed1). Averaging each side over its 2 seeds:

| dataset | fast_scnn-aug (2-seed avg) | supernet-large-aug (2-seed avg) | gap |
|---|---:|---:|---:|
| cityscapes | 0.5208 | 0.5311 | +0.0103 |
| acdc/fog | 0.5629 | 0.5633 | +0.0004 |
| acdc/night | 0.3823 | 0.3799 | **−0.0024** |
| acdc/rain | 0.5053 | 0.4933 | **−0.0120** |
| acdc/snow | 0.5073 | 0.5008 | **−0.0065** |

With a second seed on each side, the picture changes from "supernet wins 4/5" to
**supernet wins 2/5, loses 3/5** -- exactly the over-reading risk flagged above, now
confirmed. Every margin is still tiny (0.04-1.2 mIoU points, both directions) and
both sides' own seed-to-seed spread is comparable in size (supernet seed0 vs. seed2:
0.9-3.1 points; fast_scnn seed0 vs. seed1: 0.4-2.0 points) -- this is **near-parity
within noise**, not a clean win for either side. This is arguably the more scientifically
expected and more defensible result for RQ2's actual hypothesis ("subnets land within
1.0-1.5 mIoU of independent training", not "beat it") than the earlier single-seed
"wins 4/5" framing was. **Go/no-go: still passes** -- nowhere close to the >2 mIoU
no-go trigger in either direction -- but the honest headline claim is "matches
independent per-budget training at comparable cost", not "outperforms it".

## FINAL (3 seeds per side) 2026-09-14 -- confirmed near-parity, headline-ready

`pace_seg_v1_aug_seed3` and `fast_scnn_seed2_aug` finished, giving the full 3
augmented seeds per side `RESEARCH_PLAN.md` §9 rule 8 wants for a headline number.
3-seed averages:

| dataset | fast_scnn-aug (3-seed avg) | supernet-large-aug (3-seed avg) | gap |
|---|---:|---:|---:|
| cityscapes | 0.5228 | 0.5338 | +0.0110 |
| acdc/fog | 0.5629 | 0.5650 | +0.0021 |
| acdc/night | 0.3822 | 0.3757 | −0.0065 |
| acdc/rain | 0.5050 | 0.5027 | −0.0023 |
| acdc/snow | 0.5048 | 0.5055 | +0.0007 |

Supernet wins 3/5 splits, loses 2/5 -- every margin is now **≤1.1 mIoU points**,
including a near-exact tie on `acdc/snow` (+0.0007). This is the most stable estimate
so far and confirms the 2-seed reading: **true near-parity**, not a win for either
side. **Go/no-go: passes**, headline-ready with 3 seeds per side. The honest claim
for the paper is that the shared supernet matches independently-trained, same-budget
training within ~1 mIoU point across clean and all four adverse conditions, at a
fraction of the training/maintenance cost of training 4 separate models per device --
RQ2's actual hypothesis, confirmed, not exceeded.

## Side finding: augmentation does not help every architecture equally

`segformer_b0` was also re-trained with augmentation (`baseline_segformer_b0_seed0_aug`)
and, unlike `fast_scnn` (+9 to +11.5 points from augmentation), scored **lower** with
augmentation than without: Cityscapes 0.5665 (no aug) -> 0.5513 (aug), a **-0.0152**
change, with similar small drops on every ACDC condition. Not yet understood -- possible
causes (untested): SegFormer's transformer encoder may already have enough implicit
regularization (attention structure, overlap patch embedding smooths some scale
variance) that added scale/crop/color augmentation is net noise rather than signal at
this step budget; or the same random-scale-and-crop recipe simply needs different
hyperparameters (crop range, color-jitter strength) per architecture family rather
than one fixed recipe for all. `bisenetv2`, `ddrnet23_slim` and `mobilenetv3_deeplabv3`
have not been re-trained with augmentation yet, so it is unknown whether this is a
CNN-vs-transformer split or specific to SegFormer -- **do not generalize
"augmentation helps" as a blanket claim** without checking each architecture.

## Caveat before this becomes a manuscript claim

- Per-class breakdown not yet pulled for fast_scnn vs. supernet-large in the same
  table -- worth checking whether the near-parity aggregate hides a person/rider/
  vulnerable-road-user-class split the way the acdc/fog investigation found, before
  calling the two methods equivalent in every respect that matters.
- The segformer_b0 augmentation regression (above) is unexplained and should be
  investigated (e.g. per-class breakdown, or an ablation over augmentation strength)
  before writing any general "augmentation improves robustness" claim in the paper.
