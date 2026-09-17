# Ablation: with vs. without in-place distillation — v1, 2026-09-17

First real result for one of `RESEARCH_PLAN.md` §10's ten ablation axes
("with/without distillation"), previously 0% started. Trained a second copy
of the supernet with the in-place distillation loss term disabled
(`--override search.alpha_distill=0`, `pace_seg_v1_no_distill_seed0`, seed=0,
100k steps, augmentation on by default -- identical recipe to
`pace_seg_v1_aug_seed0` otherwise, which serves as the with-distillation
comparison point) and evaluated both at all 4 elasticity levels.

## Result: distillation's benefit is level-dependent, not uniform

*(delta = no-distillation mIoU − with-distillation mIoU; negative means
distillation helped)*

| level | cityscapes | acdc/fog | acdc/night | acdc/rain | acdc/snow | mean delta |
|---|---:|---:|---:|---:|---:|---:|
| tiny | +0.53 | -0.27 | +1.04 | +0.02 | +0.16 | **+0.30** |
| small | -0.47 | -0.36 | -0.40 | -1.35 | -0.74 | **-0.66** |
| medium | -0.69 | +0.68 | -0.22 | +0.48 | +1.12 | **+0.27** |
| large | +1.33 | +1.08 | -0.32 | -0.55 | +1.67 | **+0.64** |

(mIoU points, i.e. delta x 100 from the raw 0-1 scale.)

At the **`small` level**, distillation clearly helps: every one of 5 splits is
worse without it (-0.36 to -1.35 points), the only level with a fully
one-directional result. At **`tiny`/`medium`/`large`**, the picture is mixed
to slightly *negative* for distillation in this single-seed run -- more
splits favor no-distillation than favor distillation.

This is worth stating plainly because it goes against the simple expectation
that in-place distillation from the `large` (teacher) level should help every
smaller level roughly equally. It does not, at least in this one comparison.
One candidate explanation specific to `large`'s own numbers: `large` **is**
the teacher level in this design (`training/step.py`: `if level ==
teacher_level: teacher_logits = logits.detach()`), so it never receives a
distillation loss term itself -- `large`'s with/without-distillation
comparison is really testing an indirect effect (e.g. whether computing and
back-propagating the KD loss through the smaller levels changes how gradients
interact with shared stem/early-layer weights the teacher also uses), not
distillation acting on `large` directly. `tiny` and `medium`'s mixed results
have no equally direct explanation yet.

## Not yet done

- **This is n=1 per side.** A single-seed comparison is exactly the kind of
  result this project has previously found unreliable (the original
  supernet-vs-baseline go/no-go comparison needed 3 seeds before the
  direction of the result stabilized, `reports/baseline_comparison_gap_check_20260912.md`).
  Do not cite the level-dependent pattern above as confirmed until repeated
  on at least one more seed pair.
- No investigation yet into *why* `small` benefits uniquely -- is it specific
  to `small`'s position in the architecture, its parameter count, or
  something about how boundary-aware loss interacts with distillation at that
  size?
- Only `alpha_distill=0` (fully off) tested against the default `0.5` -- no
  sweep of intermediate distillation weights.
