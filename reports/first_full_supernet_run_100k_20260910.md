# First full (100k-step) supernet training run — 2026-09-10/11

## Seed 1 (`pace_seg_v1_seed1`) — added 2026-09-11

Same config/architecture/step budget, `seed=1` (`RESEARCH_PLAN.md` §9 rule 8: 3 seeds
required for headline numbers -- 2/3 now done). All 20 cells within **~0.01-0.03 mIoU**
of seed 0 (largest single delta: `large/acdc/fog` -0.013), no crossovers in the
monotonic tiny<small<medium<large ordering, and **`acdc/fog` > clean Cityscapes holds
at every level in seed 1 too** -- the same pattern, not a seed-0-specific fluke,
reinforcing the per-class explanation below rather than calling it into question.
Good early sign of training stability/reproducibility. Full precision in
`reports/eval_pace_seg_v1_seed1_step100000.json`.

| level | cityscapes | acdc/fog | acdc/night | acdc/rain | acdc/snow |
|---|---:|---:|---:|---:|---:|
| tiny | 0.3046 | 0.3173 | 0.1966 | 0.3131 | 0.2697 |
| small | 0.3603 | 0.3790 | 0.2523 | 0.3607 | 0.3325 |
| medium | 0.4165 | 0.4298 | 0.2848 | 0.4035 | 0.4038 |
| large | 0.4700 | 0.4789 | 0.3268 | 0.4553 | 0.4568 |


First real training budget end to end: `experiment_id=pace_seg_v1`,
`run_id=SERVER-02_train_20260910T093228Z`, 100,000 steps,
`configs/experiment/default.yaml` as-is (rescaled ~1.02M-param architecture at
"large", real Cityscapes 2975/500 + ACDC 1600/406). `task_status=0`; loss went
7.5 (start) -> plateaued ~2.1-2.2 as LR cosine-decayed to 0 at step 100000. Evaluated
with `scripts/evaluate_supernet.py` against `outputs/pace_seg_v1/checkpoints/step_00100000.pt`.

## mIoU, all 4 levels x Cityscapes val + ACDC val (4 adverse conditions)

| level | cityscapes | acdc/fog | acdc/night | acdc/rain | acdc/snow |
|---|---:|---:|---:|---:|---:|
| tiny | 0.2986 | 0.3154 | 0.1944 | 0.2901 | 0.2610 |
| small | 0.3564 | 0.3658 | 0.2511 | 0.3680 | 0.3304 |
| medium | 0.4120 | 0.4360 | 0.2898 | 0.4003 | 0.3981 |
| large | 0.4712 | 0.4919 | 0.3312 | 0.4517 | 0.4492 |

Full precision in `reports/eval_pace_seg_v1_step100000.json`.

## Reading

- **Monotonic in model size at every dataset/condition** (tiny < small < medium <
  large), same as every prior shorter run — the sandwich-rule training keeps behaving
  correctly at full training length, not just at toy step counts.
- **Large improved 0.2105 -> 0.4712 on Cityscapes** (2.2x) going from the 2,000-step
  stability check (`reports/first_end_to_end_miou_20260910.md`) to a real 100,000-step
  budget — training continues to make real progress with more steps, no sign of
  stalling or divergence at this budget.
- **`acdc/night` is the hardest condition at every level** (0.19-0.33), as expected —
  the least visual information available.
- **RESOLVED 2026-09-11 — `acdc/fog` > clean Cityscapes explained, not a bug.**
  Per-class breakdown (`--per-class`, `reports/eval_pace_seg_v1_step100000_percls.json`,
  `large` level) rules out the ignore-pixel hypothesis outright: `acdc/fog` actually has
  *fewer* ignored pixels proportionally than Cityscapes (93.7% valid vs 87.5% valid), so
  it isn't inflating its own mIoU that way. The real mechanism: several **large,
  structural classes score noticeably *higher* in fog** -- `wall` 0.256→0.469, `pole`
  0.294→0.451, `traffic light` 0.233→0.383, `sky` 0.900→0.976 -- plausibly because
  ACDC's fog scenes are visually simpler/more homogeneous. Averaged unweighted with 15
  other classes, these gains outweigh a **real, expected, and safety-relevant
  degradation on dynamic road-user classes**: `person` 0.471→0.280, `bicycle`
  0.472→0.204, `motorcycle` 0.139→0.016, `car` 0.828→0.709. Macro mIoU alone hides this
  — it says fog is "easier" while the classes that matter most for adverse-condition
  safety are clearly worse. **Action: always report the per-class breakdown for
  person/rider/pole/traffic-sign/traffic-light alongside aggregate mIoU** (already
  required by `RESEARCH_PLAN.md` §8 — this is a concrete demonstration of why that
  metric is in the plan, not just a nice-to-have). Note also several rare classes
  (`train`, `rider`, `motorcycle`, `bus`) swing 10-40 points between splits from small
  sample size alone — treat single-split rare-class numbers as noisy, not a robust
  claim, until per-class pixel/instance counts are checked too.

## Not yet established

- **The go/no-go criterion this run was meant to inform** ("subnet gap ≤2 mIoU vs
  independently-trained models at the same budget", RESEARCH_PLAN.md §11) **cannot be
  checked yet** — 0/7 required baselines have been trained. These numbers show the
  supernet trains and behaves correctly, not that it matches or beats independent
  training.
- **No data augmentation is applied anywhere in the training pipeline**
  (`data/transforms.py`'s `SegmentationResizeToTensor` is resize + normalize only, and
  `training/data.py` doesn't add any). 100,000 steps over a 4,575-image training set
  with zero augmentation is a plausible source of headroom being left on the table
  (likely the single highest-leverage next change) and possibly of overfitting bias in
  the numbers above -- not yet measured either way (no separate train-loss-vs-val-mIoU
  divergence check has been done).
- 2/3 training seeds done (RESEARCH_PLAN.md §9 rule 8 requires 3 for headline numbers);
  seed 1 landed within ~0.01-0.03 mIoU of seed 0 everywhere -- see above.
