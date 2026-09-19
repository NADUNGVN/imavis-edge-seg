# RQ1 budget sweep — v1, 2026-09-20

Proposed by Codex (2026-09-20 coordination round) as stronger evidence than the
single 10ms-budget snapshot in `reports/flops_baseline_v1_20260917.md`: sweep
*many* latency budgets, not one, across every (reference device the FLOPs proxy
is calibrated on) x (target device), and report mis-selection rate, accuracy
regret, and unused-budget slack/violation rate rather than a handful of
one-off examples. Built entirely locally (`src/imavis_edge_seg/search/flops.py`'s
new `evaluate_flops_proxy_at_budget`, `scripts/rq1_budget_sweep.py`) — no
training or server access needed, reuses the existing real 4-device LUT and the
existing `pace_seg_v1_aug_seed0` eval JSON. 154/154 tests pass (4 new), ruff
clean, mypy clean.

## Setup

40 log-spaced budgets from 0.75ms to 49.42ms (0.8x the global min real latency
to 1.2x the global max), x 4 reference devices (whichever device calibrates the
FLOPs→latency proxy rate) x 4 target devices (E1/hailo_hef, E2/E3/E5/tensorrt_gpu)
= 640 (reference, target, budget) evaluations, cityscapes dataset,
`pace_seg_v1_aug_seed0` (the go/no-go comparison's own FP32 checkpoint).

## Result

| reference | target | mis-selection % | mean accuracy regret | budget-violation % | mean unused slack (ms) |
|---|---|---:|---:|---:|---:|
| E1 | E1 (self) | 65.0 | −0.0313 | 65.0 | +0.27 |
| E1 | E2 | 47.5 | −0.0209 | 47.5 | +2.93 |
| E1 | E3 | 65.0 | +0.0544 | 7.5 | +9.19 |
| E1 | E5 | 42.5 | +0.0255 | 12.5 | +7.41 |
| E2 | E1 | 65.0 | −0.0313 | 65.0 | +0.19 |
| E2 | E2 (self) | 50.0 | −0.0232 | 50.0 | +2.86 |
| E2 | E3 | 62.5 | +0.0525 | 7.5 | +9.17 |
| E2 | E5 | 45.0 | +0.0236 | 15.0 | +7.37 |
| E3 | E1 | 95.0 | −0.1087 | 95.0 | **−12.11** |
| E3 | E2 | 95.0 | −0.0952 | 95.0 | **−9.66** |
| E3 | E3 (self) | 37.5 | −0.0188 | 35.0 | +6.41 |
| E3 | E5 | 67.5 | −0.0478 | 67.5 | +1.75 |
| E5 | E1 | 95.0 | −0.0762 | 95.0 | **−6.36** |
| E5 | E2 | 85.0 | −0.0606 | 85.0 | **−3.86** |
| E5 | E3 | 45.0 | +0.0092 | 22.5 | +7.69 |
| E5 | E5 (self) | 40.0 | −0.0190 | 37.5 | +4.35 |

*(mis-selection % = fraction of the 40 budgets where the proxy picks a
different level than real measured latency would; accuracy regret = real mIoU
− proxy's own REAL mIoU, averaged over budgets where both sides pick something,
positive = proxy underperforms; budget-violation % = fraction of budgets where
the proxy's chosen level's REAL latency actually exceeds the budget; mean
unused slack = average `budget − proxy's real latency`, negative = average
violation.)*

## Reading

**Even self-calibration (reference == target) fails 37.5–65% of the time.** A
FLOPs proxy calibrated on a device's *own* real data, then applied back to that
same device, still picks the wrong level on well over a third of budgets —
confirming, with far more statistical weight, the single-budget finding in
`reports/flops_baseline_v1_20260917.md` that even E3-calibrated-for-E3 disagreed
with real selection at the one 10ms budget tested there. A simple
through-origin linear fit is not just a poor cross-device transfer — it is a
poor model of one device's own latency-vs-FLOPs relationship.

**Cross-device transfer is sometimes catastrophic, not just noisy.**
E3→E1, E3→E2, E5→E1, and E5→E2 all show 85–95% mis-selection with **large
negative mean slack** (−3.86 to −12.11ms) — meaning the proxy's choice, on
average, does not merely underperform, it **actively violates the real latency
budget** most of the time. Mechanism: E3 (AGX Xavier) and E5 (Orin Nano) are
the two fastest real devices measured; a rate calibrated on either one
systematically *under*-predicts cost when applied to a slower device (E1
Hailo-8, E2 Xavier NX), so the proxy keeps picking levels it believes fit the
budget that in reality do not.

**Accuracy regret's sign is not always in the "expected" direction.** Several
cells show *negative* mean regret (proxy's real mIoU is higher, on average,
than what real selection would have picked) — but this is not a point in the
proxy's favor: those are exactly the high-budget-violation cells, where the
proxy is achieving that accuracy by silently exceeding the real latency budget,
not by making a better in-budget choice. Regret alone, without the violation
rate alongside it, would be a misleading summary statistic on its own — this is
why the sweep reports both together rather than either in isolation.

## Framing for the manuscript

Per the 2026-09-20 coordination-log framing: this is the quantitative core of
RQ1 (hardware-measured selection vs. a FLOPs-based proxy), now with real
statistical weight (640 evaluations across a full device x device grid) rather
than a handful of illustrative examples. It supports the same conclusion as
`reports/flops_baseline_v1_20260917.md`, more strongly: a FLOPs-based proxy
cannot substitute for real per-device measurement, and the failure mode is not
occasional noise but a frequent, sometimes severe (real budget violation)
miscalibration.

## Not yet done

- Only cityscapes / one FP32 checkpoint tested — not yet swept across ACDC
  conditions or other seeds.
- The through-origin linear fit is deliberately simple; a fixed-overhead +
  linear model might fit real hardware better and would be a fairer "best
  case" upper bound for what a FLOPs proxy could achieve, not yet tried.
- Does not yet convert into the specific "≥15–20% cost reduction" percentage
  the §11 go bar's phrasing anticipates — this quantifies *how often and how
  badly* the proxy is wrong, not yet a single headline percentage framed the
  same way the go bar is worded.
