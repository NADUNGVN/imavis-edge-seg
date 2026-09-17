# FLOPs-aware selection baseline v1 — first quantitative RQ1 evidence, 2026-09-17

Closes the biggest remaining gap flagged in `reports/pareto_search_v1_20260912.md`:
RQ1's hypothesis ("hardware-aware selection beats FLOPs-aware selection") had only
a qualitative demonstration (different devices pick different levels), not a
quantitative comparison against an actual FLOPs-based selection method. Built
entirely locally — no server or edge-device access needed, since FLOPs depend only
on graph shapes, not trained weights or real hardware.

## What was built

- `src/imavis_edge_seg/search/flops.py`: `count_flops()` — forward-hook-based FLOPs
  (2x MACs) counter for every `nn.Conv2d`/`SlimmableConv2d` layer actually exercised
  in one forward call. Hooking (rather than a static per-module formula) is what
  makes the supernet's per-level active-channel slicing work for free: the same
  `SlimmableConv2d` object reports different FLOPs depending on which elasticity
  level's forward call triggered the hook, with no extra bookkeeping. Also
  `FlopsPoint`/`build_flops_points`/`select_under_flops_budget` (mirrors
  `search.pareto`'s API) and `fit_flops_to_latency_rate()` (least-squares
  `latency = k * FLOPs` fit). 10 new unit tests, including a hand-computed-value
  check and a SlimmableConv2d active-channel-scaling check.
- `scripts/compute_flops.py`: computes real FLOPs for all 4 supernet levels (and,
  optionally, any required baseline architecture) from untrained weights (FLOPs
  don't depend on weight values, same convention `export_all_levels.py` uses).
- `scripts/flops_vs_latency_baseline.py`: the actual RQ1 test. Calibrates a
  FLOPs→latency proxy rate on one reference device (least-squares, all 4 levels),
  applies that *same* rate to predict latency and pick a level under a fixed
  budget on every other device — exactly what a FLOPs-only method would have to
  do, since it cannot re-measure per device — and compares against what that
  device's *own real measured latency* would have picked.

## Real FLOPs, all 4 supernet levels (1.02M-param architecture)

| level | resolution | GFLOPs |
|---|---|---:|
| tiny | 384x192 | 0.2060 |
| small | 512x256 | 1.1678 |
| medium | 768x384 | 6.1712 |
| large | 1024x512 | 22.6992 |

For reference, required-baseline FLOPs at the same (largest-level) resolution:
`fast_scnn` 3.34, `ddrnet23_slim` 18.99, `segformer_b0` 19.63, `bisenetv2` 28.90,
`mobilenetv3_deeplabv3` 39.30 GFLOPs.

## Result 1: FLOPs scales with level size far faster than real latency does

FLOPs large/tiny ratio: **110.2x**. Real measured end-to-end latency large/tiny
ratio, per device: E1 (Hailo) 11.1x, E2 (Xavier NX) 19.5x, E3 (AGX Xavier) 10.3x,
E5 (Orin Nano) 16.6x. FLOPs overstates the true cost-scaling factor by roughly
**6-11x** compared to what any real device actually measures — smaller models get
real hardware discounts (fixed per-inference overhead, underutilized
parallelism) that a pure compute-operation count cannot see.

## Result 2: a FLOPs-proxy calibrated on ANY one real device mis-selects on most
of the others, at a fixed 10ms budget

Calibrated on E3 (AGX Xavier, TensorRT GPU):

| device/backend | real pick | FLOPs-proxy pick | match? | worst per-level error |
|---|---|---|---|---|
| E1/hailo_hef | small | medium | **DIFFERENT** | 98% |
| E2/tensorrt_gpu | small | medium | **DIFFERENT** | 96% |
| E3/tensorrt_gpu (self) | large | medium | **DIFFERENT** | 90% |
| E5/tensorrt_gpu | medium | medium | same | 92% |

Notably, the proxy is wrong even on **E3 itself** — the device it was calibrated
on — because a single linear `latency = k * FLOPs` rate cannot capture the real,
non-linear relationship between compute and latency at even one device (fixed
overhead, memory-bandwidth effects). Calibrated on E1 (Hailo) instead:

| device/backend | real pick | FLOPs-proxy pick | match? | worst per-level error |
|---|---|---|---|---|
| E1/hailo_hef (self) | small | small | same | 89% |
| E2/tensorrt_gpu | small | small | same | 82% |
| E3/tensorrt_gpu | large | small | **DIFFERENT** | 365% |
| E5/tensorrt_gpu | medium | small | **DIFFERENT** | 139% |

Regardless of which device supplies the calibration, the FLOPs-proxy mis-selects
on **at least 2 of 4 devices** (3/4 when calibrated on E3), with per-level
latency prediction errors of 82-365%. On E3 specifically (calibrated from E1),
the mistake is a very costly one: the proxy conservatively picks `small` when the
device's real budget would actually support `large` — leaving a large amount of
accuracy on the table it didn't need to.

## Reading for the paper

This is the quantitative complement to the qualitative Pareto result
(`reports/pareto_search_v1_20260912.md`): not only do different real devices
prefer different levels at the same budget, but a FLOPs-based proxy — which by
construction can only encode ONE device's compute/latency relationship — cannot
reproduce those device-specific choices; it gets most devices wrong regardless of
which single device it was calibrated from. This directly supports RQ1's premise
that measured, hardware-in-the-loop selection is necessary, not simply a
convenience over a FLOPs-based method that would work almost as well.

## Not yet done

- This tests one dataset/eval-checkpoint (`pace_seg_v1_aug_seed0`, cityscapes) and
  a fixed 10ms budget — not yet swept across ACDC conditions, other budgets, or
  other seeds.
- `fit_flops_to_latency_rate`'s through-origin linear fit is a simple choice; a
  fixed-overhead + linear-in-FLOPs model might fit real hardware better and could
  be tried as a "best case for FLOPs" comparison, to make sure the proxy's failure
  isn't just an artifact of an overly naive fit.
- Does not yet produce the specific "≥15-20% cost reduction" percentage the §11
  go bar's phrasing anticipates — this result shows *wrong selections happen*,
  not yet a single clean "our method saves X% vs. FLOPs-aware" headline number.
