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

**Superseded below**: a 2nd seed pair (2026-09-17, same day) shows the
`tiny`/`medium` part of this reading did not replicate -- see the Update
section. Only the `small`-level finding held up.

## Update 2026-09-17, same day: 2nd seed pair -- the caution above was warranted,
only `small` replicates

Second pair: `pace_seg_v1_no_distill_seed3` (seed=3, 100k steps, `alpha_distill=0`)
vs. `pace_seg_v1_aug_seed3` (seed=3, distillation on) -- the exact same design as
the seed0 pair above, a different random seed.

| level | cityscapes | acdc/fog | acdc/night | acdc/rain | acdc/snow | mean delta |
|---|---:|---:|---:|---:|---:|---:|
| tiny | +0.26 | -1.61 | -0.68 | -1.77 | -0.93 | **-0.95** |
| small | -0.68 | -0.56 | -1.38 | -0.89 | -0.96 | **-0.89** |
| medium | -0.77 | -1.26 | -1.09 | -0.45 | -0.92 | **-0.90** |
| large | -0.04 | -0.27 | +1.12 | -0.66 | -0.60 | **-0.09** |

Comparing the two seeds' mean deltas per level:

| level | seed0 mean delta | seed3 mean delta | agree? |
|---|---:|---:|---|
| tiny | +0.30 | -0.95 | **NO -- opposite sign** |
| small | -0.66 | -0.89 | yes -- distillation helps both times |
| medium | +0.27 | -0.90 | **NO -- opposite sign** |
| large | +0.64 | -0.09 | roughly -- both near-neutral/small |

**Only `small` replicates**: distillation clearly helps at `small` in both
seeds (a real effect, not noise). `tiny` and `medium` flip sign entirely
between seeds -- seed0 mildly favored *not* using distillation there, seed3
clearly favors *using* it, by a similar or larger margin. This is exactly the
single-seed-comparison risk flagged as a caveat when only seed0 existed, now
confirmed to have been real: the `tiny`/`medium` "mixed, slightly negative"
finding from the first seed did **not** hold up and should not be cited.
`large` stays roughly neutral in both seeds, the one part of the original
reading that survived.

**Revised, citable claim**: in-place distillation provides a real, consistent
benefit at the `small` elasticity level (2/2 seeds, 10/10 splits favor it);
its effect at `tiny`/`medium` is unresolved (seed-dependent, opposite signs);
`large` (the teacher level, which never receives the distillation loss term
directly) shows no clear effect either way.

## Not yet done

- A 3rd seed pair would help break the tie at `tiny`/`medium`, though at this
  point the honest read is that those two levels' distillation effect is small
  relative to seed-to-seed noise, not that a 3rd seed will obviously reveal a
  hidden consistent direction.
- No investigation yet into *why* `small` specifically shows a robust,
  seed-independent benefit while `tiny`/`medium` do not.
- Only `alpha_distill=0` (fully off) tested against the default `0.5` -- no
  sweep of intermediate distillation weights.
