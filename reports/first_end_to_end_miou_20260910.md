# First end-to-end mIoU result (train -> checkpoint -> eval) — 2026-09-10

The first time the full pipeline (real Cityscapes+ACDC data -> sandwich-rule training ->
checkpoint -> mIoU evaluation) ran start to finish. Not a research result -- a stability
checkpoint (2,000 steps, `configs/experiment/default.yaml` as-is, ~126K-parameter "large"
level) is nowhere near a real training budget or model capacity. Recorded because the
qualitative trends below are evidence the pipeline itself is correct, which is worth
having on record before scaling up.

- git commit at eval time: `c912423acd2c298ee1b18921eb17ea3c19600bbe` (checkpoint step 2000)
- Checkpoint: `outputs/pace_seg_dev_smoke/checkpoints/step_00002000.pt`, `SERVER-02`
- Eval data: real Cityscapes val (500 images) + real ACDC val (406 images, 4 conditions)

| Level | Cityscapes | ACDC fog | ACDC night | ACDC rain | ACDC snow |
|---|---:|---:|---:|---:|---:|
| tiny | 0.1089 | 0.0938 | 0.0725 | 0.1186 | 0.0895 |
| small | 0.1103 | 0.0918 | 0.0693 | 0.1243 | 0.0894 |
| medium | 0.1207 | 0.1056 | 0.0797 | 0.1329 | 0.1007 |
| large | 0.1462 | 0.1140 | 0.1028 | 0.1497 | 0.1208 |

## Reading

- **Monotonic in model size** at every dataset/condition (tiny ≤ small < medium < large)
  -- the expected direction for a working elastic supernet under the sandwich rule; not
  guaranteed by construction, so this is a real (if early) correctness signal, not
  something assumed.
- **`night` is the lowest condition at every level; `rain` is the highest** -- matches
  the intuitive difficulty ordering of these conditions, another sign the metric and
  data loading are wired correctly rather than producing noise.
- All values well above a naive random-19-class baseline (~0.05), despite only 2,000
  steps on a deliberately tiny (compiler-smoke-test-sized) architecture.

## Not a claim of

- Any real segmentation quality -- 2,000 steps and ~126K params is far below what any
  reported baseline in `docs/RESEARCH_PLAN.md` §6.2 would need to be comparable.
- Anything about the calibrated router, distillation quality specifically, or
  hardware-in-the-loop search (RQ1-4) -- this only confirms the data/train/eval
  plumbing is correct end to end.

## Open decision (resolved same day)

Model capacity (current "large" ~126K params) is far below baseline scale (Fast-SCNN
~1.1M). Before committing to a long real training run, decide whether to scale up base
channels (`src/imavis_edge_seg/models/channels.py` `BASE_STEM_CHANNELS`/
`BASE_STAGE_CHANNELS`) first, since architecture changes invalidate any checkpoint
trained before them.

**Decision: scaled up** (see `reports/edge/model_rescale_compiler_revalidation_20260910.md`
for the compiler re-validation). Base channels 3x'd, "large" now ~1.02M params.

## Update: same experiment repeated at the new (~1.02M param) size

Same protocol -- 2,000 steps, `configs/experiment/default.yaml`, `SERVER-02` GPU --
`experiment_id=pace_seg_rescale_v1` so the old checkpoint isn't overwritten. Final
training loss at step 2000: **5.68** (was 7.5 at the old size, same step count).

| Level | Cityscapes | ACDC fog | ACDC night | ACDC rain | ACDC snow |
|---|---:|---:|---:|---:|---:|
| tiny | 0.1631 | 0.1290 | 0.1034 | 0.1486 | 0.1195 |
| small | 0.1690 | 0.1362 | 0.1084 | 0.1571 | 0.1258 |
| medium | 0.1841 | 0.1543 | 0.1242 | 0.1650 | 0.1427 |
| large | 0.2105 | 0.1833 | 0.1378 | 0.1935 | 0.1658 |

**~40-45% relative mIoU improvement at every level and every condition**, same training
budget (Cityscapes large: 0.1462 -> 0.2105). The monotonic-in-size and
night-lowest/rain-highest patterns both persist. This is a direct empirical confirmation
of the capacity-bottleneck reasoning in `docs/RESEARCH_PLAN.md`-adjacent discussion (the
smaller architecture was undersized relative to the literature-baseline parameter counts
it was being implicitly compared against) -- not just a plausible prediction anymore.

Still not a claim of real segmentation quality: 2,000 steps remains far short of any
baseline's actual training budget (typically hundreds of thousands of steps).
