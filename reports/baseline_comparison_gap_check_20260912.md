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

## Caveat before this becomes a manuscript claim

- Only 2/3 supernet seeds and 1/1 fast_scnn seed so far -- `RESEARCH_PLAN.md` §9 rule 8
  wants 3 seeds for headline numbers. The margin here (2.6-6.4 points) is comfortably
  larger than the seed-to-seed noise already observed (~0.01-0.03 points), so this
  result is very unlikely to flip, but the third seed (both supernet and fast_scnn)
  should still be run before this goes in the paper.
- fast_scnn here has no augmentation either, so the comparison is fair as run --  but
  neither number reflects what either model could reach *with* augmentation (now
  available, `data.transforms.SegmentationTrainAugment`). Future re-runs of both should
  use it, and this comparison should be redone once they do.
- Per-class breakdown not yet pulled for fast_scnn's supernet-large counterpart in the
  same table -- worth checking whether the win is broad-based or concentrated in a few
  classes, same diligence as the acdc/fog investigation.
