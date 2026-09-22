# Router progressive ablation A→B→C→D — v1, 2026-09-21

Full run of the locked progressive-ablation protocol (`docs/COORDINATION_LOG.md` open
thread #2, design + grid locked with Codex 2026-09-20/21) on real hardware latency
data across all 4 benchmarked devices (E1/Hailo-8, E2/Xavier NX, E3/AGX Xavier,
E5/Orin Nano), all 5 evaluation splits (Cityscapes val + 4 ACDC conditions), from
`pace_seg_v1_aug_seed0`'s FP32 checkpoint. Raw per-split JSON:
`reports/router_{E1,E2,E3,E5}_20260921.json`.

## Setup recap

- **A** (`calibrated_risk`): existing single-probe-risk baseline, rank-step escalation.
- **B** (`latency_spacing_risk`): same risk signal, escalation targets a latency
  *magnitude* (closest-match) instead of a rank step.
- **C** (`candidate_specific_risk`): per-level calibrators from a shared probe, no
  explicit latency budget.
- **D** (`risk_latency_constrained`): the full method — per-candidate predicted risk
  *and* an explicit per-device latency budget, 4-step decision rule (see
  `router/policy.py`).
- **Grids**: risk-target = 5 quantiles (Q10/25/50/75/90) of fit-half per-image
  observed error, macro-averaged across all 5 splits: `[0.0662, 0.0879, 0.1202,
  0.1658, 0.2174]`. Entropy threshold grid computed the same way on raw probe score:
  `[0.396, 0.4511, 0.5098, 0.5655, 0.6475]`. Both identical across all 4 devices
  (device-independent, as designed). Device-budget grid = that device's own 4
  per-level p95 latencies (real, not log-spaced).
- **Fair comparison protocol**: A/B/C/entropy's per-budget "representative point" is
  chosen using `select_budget_matched_operating_point` on **fit-half statistics
  only**, before held-out is touched. Oracle is per-image, budget-matched, from
  ground-truth observed error. 80 (device × split × budget) comparison cells total.

One real operational incident found and fixed along the way: `outputs/` is
`.gitignore`d (never synced across machines), and E2/E5's real trtexec latency rows
(added 2026-09-16, run "directly from the dev machine" per README Phase 3) had never
actually reached the shared-NFS `outputs/benchmark_lookup_table.csv` that
`evaluate_router.py` reads on the training servers — the first E2/E5 runs silently
fell back to a synthetic 1/2/4/8ms placeholder table (the script warns about this,
which is how it was caught) rather than the real 2.166/5.003/15.172/40.548ms
(E2) / 1.088/2.667/7.844/18.334ms (E5) figures. Fixed by appending the missing rows
from the dev machine's copy directly onto the server's file; both devices then
re-ran cleanly with real latency data. **Lesson, recorded for the coordination log**:
any latency data generated outside the normal server-job path needs an explicit sync
step (or a committed `reports/edge/*.md` as the source of truth), since `outputs/`
being gitignored means "shared NFS" isn't actually guaranteed shared across every
machine that touches it.

## Headline: D vs. A, all 80 cells

| comparison | wins | ties | losses |
|---|---|---|---|
| D vs. A, all 80 cells (raw) | 40 | 12 | 28 |
| **D vs. A, 52 cells where A never violates its budget** | **36 (69%)** | **12 (23%)** | **4 (8%)** |
| D vs. A, 28 cells where A violates its budget | 4 | 0 | 24 |

The raw 40/28/12 split understates D's real advantage: **24 of A's 28 "wins" over D
only happen because A is allowed to exceed the stated latency budget** (mean
violation rate 25% among those cells, i.e. a quarter of images routed past the
budget) — not a fair comparison. Restricted to the 52 cells where A's chosen
operating point is genuinely honest about the budget, **D wins 69% of the time, ties
23%, and loses only 8%** — and every one of those 4 losses is the *same* pattern
(`acdc/rain`, one specific mid-range budget) reproduced identically across all 4
devices, with D using *less* latency than A each time (a real, small, consistent
quality/latency tradeoff on that specific split, not noise or a bug). Mean mIoU gap
on the fair comparison: **D − A = +0.0224** (vs. D − oracle = −0.0053, i.e. D tracks
the ground-truth-optimal oracle within about half a point on average).

**Budget reliability, the structural point**: D has **zero** budget violations
across all 80 cells, on any device, any split, any budget. A violates in 28/80
(35%); candidate_specific_risk (C, no budget awareness at all) violates in 44/80
(55%, worst of any strategy); entropy in 20/80 (25%); latency_spacing_risk (B) is
the best-behaved of the non-D strategies at 5/80 (6%), consistent with B already
reasoning about latency magnitude even without a hard budget.

## Per-strategy summary, all 80 cells

| strategy | mean achieved mIoU | mean violation rate | cells with violation | cells infeasible (fit-half) |
|---|---|---|---|---|
| A: `calibrated_risk` | 0.3558 | 0.0877 | 28/80 | 16/80 |
| B: `latency_spacing_risk` | 0.3196 | 0.0225 | 5/80 | 5/80 |
| C: `candidate_specific_risk` | 0.3819 | 0.2127 | 44/80 | 20/80 |
| entropy | 0.3213 | 0.0614 | 20/80 | 12/80 |
| **D: `risk_latency_constrained`** | **0.3613** | **0.0000** | **0/80** | **0/80** |

C has the highest raw mIoU, but it's the least reliable by far (a majority of its
operating points overspend their nominal budget, by construction — it has no budget
term at all). **D gets within 0.02 mIoU of C's raw quality while providing a budget
guarantee none of A/B/C/entropy can offer.** This is the central result: D isn't
just "usually better than A" — it converts an unreliable risk-only policy into one
with a hard latency contract, at a small, well-characterized quality cost relative
to the unconstrained-risk upper bound (C).

## Diagnostics

- **`prediction_inversion_rate`** (cell C's calibrators predicting a *larger*
  candidate as worse than a smaller one, adjacent-pair basis): mean **2.13%** across
  the 20 (device × split) runs, range 0.00–10.67%. Low and consistent with a
  reasonably well-behaved per-level risk signal on a real trained model (contrast
  with the 100% inversion rate seen on the untrained-model CPU smoke test used to
  verify the pipeline before this real run — confirms the diagnostic is doing its
  job, not just always reporting near-zero).
- **`probe_signal_aurc`** (the probe's own raw risk score vs. its own observed
  error, independent of any routing policy): mean **0.1467** across splits, lowest
  on `cityscapes` (0.0989, the largest/cleanest split) and highest on `acdc/night`
  (0.2616, the hardest condition) — the risk signal itself is least informative
  exactly where the router needs it most, a real limitation worth carrying into the
  paper's discussion, not just background detail.

## Reading against Codex's locked go/stop criterion

*"D only counts as a win if it improves the Pareto frontier, or reduces latency at
equal quality/risk, on a majority of devices."* On the fair (non-violating)
comparison, D wins on a clear majority of cells (69%) uniformly across all 4
devices (each device's own win/loss/tie split is identical: 10 wins / 7 losses / 3
ties out of 20 raw cells per device, 36/4/12 fair-only aggregated — the pattern does
not depend on which device is targeted, since the risk-target grid is
device-independent and only the latency numbers scale). **D is a win by this
criterion.**

## Update, 2026-09-22: 3-seed replication — CONFIRMED, stronger than seed0 alone

Per Codex's decision (D is GO, router promoted to conditional flagship pending
3-seed replication + real overhead measurement — `docs/COORDINATION_LOG.md`), A and
D were rerun on all 4 devices for `pace_seg_v1_seed2` and `pace_seg_v1_aug_seed3`
(the project's other two independently-trained supernet seeds), with the exact same
split/features/calibrator/quantile grid/LUT/metric as seed0, calibrators fit fresh
on each seed's own fit-half, **no policy changes made after seeing either seed's
result**. Raw data: `reports/router_{E1,E2,E3,E5}_{seed2,seed3}_20260922.json`.

| seed | fair cells | D wins | D losses | D ties | win+tie rate | mean (D−A), fair | D budget violations |
|---|---|---|---|---|---|---|---|
| seed0 | 52 | 36 (69%) | 4 (8%) | 12 (23%) | 92% | +0.0224 | 0/80 |
| seed2 | 44 | 36 (82%) | **0 (0%)** | 8 (18%) | **100%** | +0.0252 | 0/80 |
| seed3 | 54 | 46 (85%) | **0 (0%)** | 8 (15%) | **100%** | +0.0290 | 0/80 |

**Macro-average across 3 seeds (equal weight per seed, per Codex's rule — not a
pooled cell count): (D−A) = +0.0256** on fair cells (+0.0073 pooling all 80 cells
per seed including A's own budget-violating operating points). **All 4 of Codex's
confirmation criteria are met, with margin**:

1. D still zero hard-budget violations — **0/80 on every one of the 3 seeds.**
2. D wins or is Pareto-non-inferior to A on ≥2/3 seeds — **3/3 seeds**, at 92%,
   100%, and 100% win+tie rates respectively.
3. The 3-seed macro-average (D−A) is non-negative at the locked comparison points —
   **+0.0256, comfortably positive.**
4. No single seed shows a large, systematic regression across most
   conditions/devices — **seed2 and seed3 show *zero* fair-comparison losses for D
   at all**; only seed0 has any losses (the 4 `acdc/rain` cells discussed above),
   and even those are small and confined to one split.

**A genuinely new finding from this replication, not visible from seed0 alone**:
the `acdc/rain`-specific loss pattern that recurred identically across all 4 devices
on seed0 **does not reproduce on seed2 or seed3** — searching all fair cells on both
replication seeds finds zero D losses anywhere, not just on `acdc/rain`. This
suggests the seed0 pattern is more likely a **seed0-specific calibration quirk**
than a systematic weakness of the router design itself, which somewhat changes the
bounded-diagnostic framing below (still worth doing, but the evidence for it being a
*general* limitation is now weaker than it looked with one seed).

**Admissible claim, per Codex's pre-registered wording**: *"Candidate-specific,
device-conditioned routing consistently improves the risk–latency trade-off over
single-probe rank-based routing across three independently trained supernets."*
Not "reliable" — `acdc/night`'s weak `probe_signal_aurc` (0.2616, seed0; not yet
re-checked on seed2/seed3) remains an open limitation.

## Not yet done

- **Router runtime overhead** (probe forward pass + calibrator/decision computation
  + candidate inference + engine-switch cost, on real E1/E3 hardware) — mandatory
  closing requirement #2 per Codex, not yet started. Current latency figures are
  candidate-only from the LUT, not the router's own cost; D's advantage must survive
  adding this overhead before the router contribution is closed.
- **`acdc/rain` bounded diagnostic** (mIoU/latency delta, routing distribution,
  per-level calibration residuals, oracle regret/non-monotonicity rate) — not yet
  done. Scope note updated after the 3-seed replication: since the pattern is
  seed0-specific (doesn't reproduce on seed2/seed3), this is now a lower-priority,
  narrower diagnostic than it looked after seed0 alone — still worth doing per
  Codex's instruction, but not evidence of a general router weakness.
- `acdc/night`'s weak `probe_signal_aurc` (0.2616, seed0 only) has not been checked
  on seed2/seed3 — worth confirming it's a stable, condition-specific limitation
  rather than seed noise, given how much the `acdc/rain` pattern turned out to be
  seed-specific.
- Temporal-window routing, UIoU — deferred behind the two closing requirements
  above (overhead measurement takes priority); UIoU additionally gated on an
  annotation-feasibility audit of ACDC (no pseudo-labels).
