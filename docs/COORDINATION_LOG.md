# Claude–Codex coordination log

> Not a status report (that's `README.md`). This file exists to prevent two
> concrete, already-experienced failure modes: (1) two agents claiming the same
> `experiment_id` on the shared NFS `outputs/` across `SERVER-01..05`, silently
> overwriting each other's checkpoints; (2) duplicated or conflicting code edits
> to the same module. Update this file *before* claiming an experiment_id or
> starting a body of work that touches files another agent might also touch.
> Append-only in spirit — add new entries, don't rewrite history.

## Standing rules (agreed 2026-09-20)

- **Claude**: coordination log, code, server jobs (launched by the user pasting
  Claude's exact commands — no agent has direct SSH to `SERVER-01..05`), QAT/router
  experiments, README/report discipline (every real result gets written into
  `README.md` + a dated `reports/*.md` file, including unfavorable ones — no
  averaging away a bad result, no calling a fix "done" before a real run confirms
  it).
- **Codex**: systematic literature matrix, Related Work section, closest-work
  comparison, claim/leakage review (checking manuscript claims against what the
  repo's actual evidence supports), negative-result framing review.
- **User**: intermediary between Claude and Codex, confirms real-time server
  availability, relays command output back to Claude.
- Codex does not modify code, `README.md`, or launch/claim an experiment before
  the claim is recorded here first.
- `outputs/` is a **shared NFS mount** across `SERVER-01..05` (`docs/INFRA_OVERRIDE.md`).
  Any new training/eval run must use an `experiment_id` not already present in
  `reports/server/*.md` or in the "Active/claimed experiment_ids" table below.
  This is not a formality — four real bugs already happened this way this session
  (a baseline checkpoint silently overwritten by a same-named concurrent run; a
  status script reading the wrong server's job through a shared pointer file,
  since fixed in `scripts/server/status_train_supernet.sh`; the same shared-pointer
  bug class recurring 2026-09-20 in the brand-new
  `scripts/server/{start,status}_train_exported_subnet.sh` — keyed the "latest run"
  pointer by level only, not by (level, hostname), so cell 4's launch on SERVER-04
  could silently overwrite cell 3's pointer on SERVER-03, making a bare status
  command on either host unreliable once both used `--level large`; fixed the same
  way as `status_train_supernet.sh`, plus an explicit `<experiment_id>` search mode
  that doesn't depend on the pointer file at all. Whether cell 3's actual training
  job ran or never started is still being diagnosed separately, see below; **and
  2026-09-21: `outputs/` itself is `.gitignore`d, so it does NOT actually sync
  across every machine that touches it** — despite being called a "shared NFS
  mount", `outputs/benchmark_lookup_table.csv`'s E2/E5 rows (generated on the dev
  machine per README Phase 3, "directly via SSH from the dev machine") had never
  reached the training servers' copy of that file; the first router-evaluation
  runs against E2/E5 silently fell back to a synthetic placeholder latency table
  (caught only because `evaluate_router.py` warns on a missing lookup-table row —
  fixed by manually appending the missing rows to the server's file). **Lesson**:
  any latency/benchmark data generated outside the normal server-job path needs an
  explicit sync step, or ideally a committed `reports/edge/*.md` write-up as the
  actual source of truth, since a `.gitignore`d `outputs/` file can silently
  diverge between machines with no error until something reads a missing row).

## Active / claimed experiment_ids

| experiment_id | owner | purpose | status |
|---|---|---|---|
| `pace_seg_v1_qat_calibrated_ema_percentile_seed0` | Claude | QAT 2x2 screen cell 2: shared supernet x EMA/percentile observer, `large`, seed0 | **done** — worst-case −0.97, see below |
| `qat_exported_large_dynamic_seed0` | Claude | QAT 2x2 screen cell 3: exported-subnet x dynamic range, `large`, seed0 | **done** (first attempt failed with CUDA OOM on SERVER-03, no checkpoint written; retried successfully) — worst-case −1.55, see below |
| `qat_exported_large_ema_percentile_seed0` | Claude | QAT 2x2 screen cell 4: exported-subnet x EMA/percentile observer, `large`, seed0 | **done** — worst-case −0.70 (best cell), see below |
| `qat_exported_large_ema_percentile_seed3` | Claude | QAT 2x2 screen: seed3 confirmation of the best cell (4) | **done — CONFIRMED**, worst-case −0.51, mean −0.32, see below |
| `pace_seg_v1_qat_calibrated_ema_percentile_seed3` | Claude | Mechanistic replication: shared supernet x ema_percentile, seed3 — does the fix rescue the shared model too? | **done — yes, also passes**, worst-case −0.86, mean −0.50, see below |
| `pace_seg_v1_qat_calibrated_ema_percentile_seed2` | Claude | Codex's closing step 1: shared x ema_percentile on `pace_seg_v1_seed2`, the seed dynamic-QAT failed worst on (−2.54) — paired 3/3-seed confirmation attempt | **done — CONFIRMED**, worst-case −1.38 (large), passes at every level, see below. **QAT is now FROZEN.** |

## Agreed thesis framing (2026-09-20)

Three-tier thesis, router as flagship:
1. **RQ2** — shared supernet near-parity with independent same-budget training
   (3-seed, confirmed, `reports/baseline_comparison_gap_check_20260912.md`).
2. **RQ1** — hardware-*measured* subnet selection beats a FLOPs-based proxy
   (`reports/flops_baseline_v1_20260917.md`; RQ1 budget-sweep extension in
   progress, see below).
3. **RQ3 / flagship** — candidate-specific, device-conditioned calibrated risk
   routing. Current router (`reports/router_v1_20260914.md`) is an MVP: single
   risk estimate per image (not per candidate level), escalation uses latency
   *rank* only (not magnitude). Both are being addressed — see Open threads.

QAT is downgraded to a **bounded rescue attempt + failure analysis**, not a
required capability for the paper's central claim. Precedent: DLA was already
demoted from primary to secondary ablation (2026-09-10) after hitting an
unfixable hardware limit — same logic applies if the QAT rescue below doesn't
close the gap.

**Do not describe "calibration makes QAT worse than dynamic" as an independent
novelty claim.** Shared-quantizer/range instability in quantized supernets is
already known in the literature (per Codex). It is legitimate, valuable failure
analysis; it becomes a method contribution only if a fix is found and confirmed.

**Title, tentative**: "PACE-Seg: Hardware-Measured Elastic Semantic Segmentation
with Calibrated Risk Routing under Adverse Conditions." Drop "reliable" from the
title until UIoU/AURC/temporal-window/external-shift evidence exists; use
"risk-aware" in the interim.

## Open threads (owner: Claude unless noted)

1. **QAT rescue — 2×2 factorial screening.** All 4 cells done on seed0, `large`
   level (`reports/qat_rescue_2x2_screen_infra_20260920.md`). Worst-case mIoU
   delta vs. FP32: **cell 1 (shared×dynamic) −1.66**; **cell 2
   (shared×ema_percentile) −0.97**; **cell 3 (exported×dynamic) −1.55**; **cell 4
   (exported×ema_percentile) −0.70**. Cells 2 and 4 pass the §11 go bar
   comfortably; cell 3 sits right at the edge; cell 1 fails it (matches the
   already-known 3-seed result). **Mean-across-5-splits view (added after
   Codex's review, more precise than worst-case alone)**: cell1 −0.54, cell2
   −0.58, cell3 −0.49, cell4 −0.28 — on the **shared** model, ema_percentile's
   *mean* is not better than dynamic's (−0.58 vs. −0.54) even though its
   worst-case is much better; it **stabilizes worst-case degradation across
   conditions, not average accuracy**. Codex's review also flagged that reading
   the worst-case deltas (+0.69/+0.85 observer, +0.11/+0.27 weight-sharing) as
   clean additive factorial main effects overstates what a `max`-based metric
   supports (the "worst" split can differ between cells) — the
   observer-dominates reading still holds as a predefined-metric result, just
   not as decomposable per-factor point contributions. **Codex's assessment:
   conditional go, not solved.** Decision rule (worst-case ≤1.5 confirms;
   1.5–2.0 supporting/borderline only; >2.0 stops the rescue).

   **Cell 4 seed3 confirmation: done, CONFIRMED** — worst-case −0.51, mean
   −0.32 (seed0 was −0.70/−0.28) — both seeds pass the ≤1.5 bar comfortably,
   reproducible across seeds. **The exported-subnet × ema_percentile QAT path
   is confirmed**, subject to the claim boundary below.

   **Mechanistic replication (cell 2, shared × ema_percentile, seed3): done —
   also passes.** Worst-case −0.86, mean −0.50 (seed0 was −0.97/−0.58) — both
   seeds of the **shared** model pass the ≤1.5 bar comfortably too (both well
   inside even the strict 1.0 lower bound). On the identical seed3 checkpoint,
   cell 1 (shared×dynamic) was −1.39/−0.69, so the observer switch improves
   the shared model's worst-case by +0.53 and mean by +0.19 on this seed
   (seed0 showed a flat/slightly-worse mean effect — inconsistent across the
   2 seeds so far, worst-case improvement is the consistent part). **This
   result is stronger than the original claim boundary allowed for**: the
   shared supernet, unchanged except for the observer, now reliably passes
   the go bar on both tested seeds — evidence that `ema_percentile` fixes QAT
   accuracy regardless of shared-vs-exported weights, which weakens the case
   that weight-independence itself was doing meaningful work. Full numbers
   and the seed0-vs-seed3 mean inconsistency: `reports/qat_rescue_2x2_screen_
   infra_20260920.md`. **This full result set (not just cell 4) is handed to
   Codex for the final analysis and go/stop call** — not treated here as a
   settled "shared-supernet QAT is solved" conclusion, per the same caution
   already applied throughout this thread.

   **Codex's final decision (2026-09-20): QAT is "go with limits," then
   freeze.** Claim boundary widened to: *"At the `large` elasticity level,
   EMA-percentile activation calibration rescues shared-supernet
   fake-quantized QAT on both tested seeds, keeping worst per-condition
   degradation below one mIoU point."* Still not "shared-supernet QAT is
   solved" or "reliable INT8 deployment." Two mechanistic conclusions: (1)
   ema_percentile alone is sufficient to pass the bar on the shared model
   (seed0/seed3) — its consistent effect is reducing worst-case/tail
   degradation, not uniformly improving mean accuracy; (2) export
   specialization still adds a consistent secondary benefit (+0.18 to +0.30
   mean, +0.27 to +0.35 worst-case across the 2 seeds) — extraction isn't
   meaningless, just not necessary to pass, and it carries its own separate
   fine-tuning cost.

   **Two closing steps before freezing QAT (do these, then stop regardless of
   outcome — no further observer/hyperparameter changes after):**
   1. **Done — paired 3/3-seed confirmation at `large`.** Cell 2 on
      `pace_seg_v1_seed2` (the dynamic-QAT seed that failed worst, −2.54):
      worst-case −1.38 at `large` (also passes at every other level: tiny
      −0.69, small −0.33, medium −0.49). All 3 originally-tested supernet
      seeds now pass under `ema_percentile` (seed0 −0.97, seed3 −0.86, seed2
      −1.38), where 2 of 3 failed under `dynamic`. Largest single-seed
      observer improvement seen in the whole screen: +1.16 points on this
      seed/level (−2.54 → −1.38).
   2. **Done — rescue generalizes across the whole elastic family, not just
      `large`.** No new training: evaluated the existing seed0/seed3
      checkpoints at all 4 levels (`evaluate_supernet.py --qat`, no `--level`
      restriction), against the already-existing FP32 references.

      | level | seed0 worst-case / mean | seed3 worst-case / mean |
      |---|---|---|
      | tiny | −0.32 / +0.06 | −0.36 / +0.00 |
      | small | −0.97 / −0.51 | −0.33 / −0.15 |
      | medium | −0.51 / −0.30 | −0.56 / −0.34 |
      | large | −0.97 / −0.58 | −0.86 / −0.51 |

      Every level, both seeds, passes the ≤1.5 bar comfortably — most cells
      are well inside even the strict 1.0 lower bound; `tiny` is nearly a
      wash (mean ≈ 0). `large`'s numbers reproduce the earlier cell 2 result
      within rounding, confirming this is the same checkpoint/calibration.
      One operational note: the first eval attempt ran on the same host
      (SERVER-03) simultaneously training step 1's job, hit a CUDA OOM (not a
      code issue), resolved by re-running on a free server via shared NFS.

   **Both closing steps done, 2026-09-21 — QAT is now FROZEN**, per the
   agreed plan: no further observer/hyperparameter search. Effort moves to
   the router (open threads #2/#3 below). Final claim boundary (unchanged
   in substance from Codex's decision, numbers now complete): *"EMA-percentile
   activation calibration rescues shared-supernet fake-quantized QAT on all
   3 tested seeds and all 4 elasticity levels, keeping worst per-condition
   degradation below 1.5 mIoU points (below 1.0 in most cases)."* Still not
   "shared-supernet QAT is solved" or "reliable INT8 deployment" — no INT8
   headline claim before real compiled-engine (TensorRT/Hailo)
   accuracy+latency numbers exist (everything so far is PyTorch
   fake-quantization simulation). The outlier-sensitivity explanation
   remains consistent with the data, not directly demonstrated (would need
   activation max/percentile diagnostics) — left as genuine future work, not
   blocking. Full numbers: `reports/qat_rescue_2x2_screen_infra_20260920.md`.
2. **Router progressive ablation A→B→C→D (design locked with Codex
   2026-09-21, now that QAT thread #1 is closed — effort moves here).**
   Renamed from "2×2" to **progressive ablation** on Codex's explicit
   correction: D needs an extra axis (an explicit hardware latency budget)
   that A/B/C don't have, so it isn't a clean 2×2 factorial — call it that
   instead, don't force the 2×2 label.

   - **A: single-probe risk × rank-only cost** — existing baseline
     (`reports/router_v1_20260914.md`), unchanged.

   - **B: single-probe risk, latency-spacing-aware escalation** (not
     "latency-value cost" — Codex's exact naming, since this is still a
     heuristic ablation on using latency *magnitude* instead of *rank*, not a
     real latency optimizer). Locked rule: `r = max(1, risk / risk_target)`,
     `t_target = clip(t_min * r, t_min, t_max)` (`t_min`/`t_max` = the
     cheapest/most-expensive candidate's latency), then pick the candidate
     whose latency is **closest** to `t_target` (not "cheapest ≥ target" —
     closest-match is easier to defend and avoids jumping to an
     unusually-expensive candidate). Tie-break: prefer the cheaper candidate.

   - **C: candidate-specific risk calibration from a shared probe, rank-only
     cost** (Codex's exact naming — NOT "candidate-specific *sensing*": all
     candidates still share the one cheap probe signal, only the calibration
     mapping differs per candidate). Fit **N independent `RiskCalibrator`s**,
     one per elasticity level, on the fit-half only: same raw probe score as
     input for all N, calibrator `i`'s *target* is candidate `i`'s own
     observed error (fit-half has ground truth for every level, not just the
     probe level, so this needs no new data — just running every level's
     forward pass on the fit-half images too, which the test-half loop
     already does). Same calibrator type/features/hyperparameters
     (`num_bins`, etc.) across all four — no per-level tuning. No held-out
     tuning. Do **not** force predicted risk to be monotonically increasing
     with level size (a bigger candidate isn't guaranteed better on every
     image) — instead **report the prediction-inversion rate** (how often a
     larger candidate's predicted error exceeds a smaller candidate's) as a
     diagnostic, not something to correct. Decision rule: sort candidates by
     latency ascending, pick the first whose own calibrator predicts error ≤
     `risk_target`; if none qualify, fall back to the most expensive (safest)
     candidate — this fallback wasn't explicitly specified by Codex, chosen
     to mirror cell A's existing escalate-to-largest-on-no-match behavior;
     flagged for Codex to confirm or correct.

   - **D: risk-and-latency-constrained policy** (the actual full method —
     Codex rewrote this cell after flagging that the original "candidate-
     specific risk filtered, then cheapest" design was **operationally
     identical to C** whenever any candidate met the risk target, which isn't
     a real second ablation axis). Needs an explicit per-device hardware
     **latency budget `B`**, evaluated alongside a pre-registered grid of
     `(B, risk_target τ)` pairs — the grid must be **locked using fit-half
     only, before the held-out half is read**, same pre-registration
     discipline as the QAT screen's decision thresholds. Let `S_B = {i :
     latency_i ≤ B}` (candidates within budget) and `ê_i` = candidate `i`'s
     predicted error from its own calibrator (from C):
     1. If any candidate in `S_B` has `ê_i ≤ τ`: pick the lowest-latency one
        among those.
     2. Else (no in-budget candidate meets `τ`): pick the in-budget candidate
        with the lowest `ê_i`.
     3. Tie-break: lower latency.
     4. If `S_B` is empty (nothing fits the budget at all): pick the cheapest
        candidate overall.
     **Explicitly rejected**: a quality-gain-per-ms fallback (score depends
     on the starting point, can pick a Pareto-dominated candidate, hard to
     justify to a reviewer) — do not implement this.
     Formal read: *"Minimize measured device latency subject to calibrated
     risk and latency constraints; when the requested risk is infeasible,
     minimize predicted risk within the hardware budget."*

   **Baselines**: entropy, cell A (current calibrated rank-only router),
   oracle, and static. Report **all 4 static levels** (tiny/small/medium/
   large), not just small/large — free (no new training), and they anchor
   the Pareto frontier. **Oracle must be budget-matched and per-image**: for
   each image *and* device budget, pick the candidate with the lowest
   *ground-truth* observed error within that budget — not the old
   fixed-per-split "whichever level has the best aggregate mIoU" pick
   (unconstrained-latency oracle is explicitly disallowed now).

   **Locked evaluation protocol**: same checkpoint, split, and device LUT for
   A/B/C/D. Fit-half fits every calibrator and locks every threshold/grid
   (including D's `(B, τ)` grid) before the held-out half is touched;
   held-out half is evaluation-only. Report: quality–latency frontier,
   risk-at-coverage/AURC, budget-violation rate, routing distribution, and
   per-device results — **mIoU-per-ms is explicitly NOT the headline
   metric** (the earlier router_v1 report's mixed 2/7 efficiency result is
   exactly why). **D only counts as a win if it improves the Pareto frontier,
   or reduces latency at equal quality/risk, on a majority of devices** — not
   on aggregate/average alone.

   **Decision, from Codex**: B go, C go, D go (with the corrected
   risk-and-latency-constrained design above — no joint/hierarchical
   calibrator this round). **Only proceed to real router-overhead
   measurement, temporal-window routing, and UIoU (item 5 below) if D shows
   that Pareto/equal-risk-latency advantage** — otherwise those stay
   deferred.

   **Implementation: done and verified 2026-09-21, not yet run on a real
   device.** `router/policy.py` (B/C/D decision functions),
   `router/calibrator.py::fit_per_level_calibrators`/
   `prediction_inversion_rate`, `router/selective_metrics.py` (AURC,
   risk-at-coverage, budget-violation rate), `router/grid.py`
   (`macro_quantile_grid`, `select_budget_matched_operating_point`) — all
   built exactly to Codex's locked spec, 33 new tests covering the exact
   worked examples from that spec. `scripts/evaluate_router.py` fully
   rewritten: both fit/held-out halves now cache every level's (pred, mask)
   once (no re-inference for any grid point); the risk-target and
   entropy-threshold grids are macro-averaged across every split's fit-half
   before any held-out data is touched; A/B/C/entropy report a full
   quality-latency frontier across the shared grid; D reports its full 4×5
   grid; a budget-matched representative point is selected per strategy per
   device budget from fit-half statistics only; oracle is redesigned to be
   budget-matched and per-image (no more `--eval-json`); static reports all
   4 levels. Verified end to end with a manual CPU smoke test (untrained
   supernet, synthetic fake Cityscapes data, synthetic lookup table) —
   correct grid shapes, correct budget-boundary behavior (oracle at the
   tightest budget exactly matches `static_tiny`), valid JSON output.
   211/211 tests pass, ruff clean, mypy clean.

   **Real run: done 2026-09-21, all 4 devices, D is a confirmed win.**
   `reports/router_progressive_ablation_v1_20260921.md`, raw data
   `reports/router_{E1,E2,E3,E5}_20260921.json`. Headline: on the 52/80
   cells where baseline A honestly respects its stated budget, **D wins
   69%, ties 23%, loses 8%** (mean mIoU gap +0.0224); the raw 40/28/12
   split understates this because 24 of A's 28 "wins" only happen when A
   itself violates the budget (mean 25% of images over budget in those
   cells). **D has zero budget violations across all 80 cells** — A
   violates in 35%, C (candidate-specific, no budget term) in 55%, entropy
   in 25%, B in 6%. C has the highest raw mIoU (0.3819) but is the least
   reliable; D gets within 0.02 mIoU of C while providing a hard latency
   guarantee none of A/B/C/entropy have. Diagnostics: mean prediction-
   inversion rate 2.13% (low, real signal, contrasts with the untrained
   smoke test's 100%); mean probe-signal AURC 0.1467, worst on
   `acdc/night` (0.2616) — the risk signal is least informative exactly
   where routing matters most, a real limitation to carry into the paper.
   **Per Codex's locked go/stop criterion ("D wins if it improves Pareto
   or reduces latency at equal quality/risk on a majority of devices"): D
   wins, identically across all 4 devices** (the risk-target grid is
   device-independent, only latency scales). One real operational
   incident: `outputs/` is gitignored and never syncs across machines —
   E2/E5's real latency rows (generated on the dev machine, per README
   Phase 3) had never reached the servers' shared NFS copy, so the first
   E2/E5 runs silently used a synthetic placeholder table until the
   `evaluate_router.py` warning caught it; fixed by manually appending the
   missing rows.

   **Codex's decision (2026-09-21): D is GO, router becomes flagship
   *conditional* on two closing requirements.** Explicit caution: the 80
   cells are not 80 independent statistical observations (τ levels,
   conditions, and devices are correlated; the 4 identical `acdc/rain`
   losses across devices are likely the *same* underlying failure mode,
   not 4 independent replications) — "consistent across 4 devices" is
   evidence of LUT-portability, not 4 independent accuracy replications.

   **Closing requirement 1 — 3-seed replication: DONE 2026-09-22, CONFIRMED,
   stronger than seed0 alone.** A and D rerun on all 4 devices for
   `pace_seg_v1_seed2` and `pace_seg_v1_aug_seed3` (locked protocol
   unchanged: same split/features/calibrator/quantile grid/LUT/metric;
   each seed's calibrator fit only on its own fit-half; no policy changes
   after seeing either result). Raw data: `reports/router_{E1,E2,E3,E5}_
   {seed2,seed3}_20260922.json`; full writeup
   `reports/router_progressive_ablation_v1_20260921.md`'s 2026-09-22
   update.

   | seed | fair cells | D wins | D losses | win+tie rate | mean (D−A) |
   |---|---|---|---|---|---|
   | seed0 | 52 | 36 (69%) | 4 (8%) | 92% | +0.0224 |
   | seed2 | 44 | 36 (82%) | 0 (0%) | 100% | +0.0252 |
   | seed3 | 54 | 46 (85%) | 0 (0%) | 100% | +0.0290 |

   **Macro-average across 3 seeds (equal weight per seed): (D−A) = +0.0256**
   fair cells (+0.0073 pooling all 80). **All 4 confirmation criteria met,
   with margin**: (1) D still zero hard-budget violations, 0/80 on every
   seed; (2) D wins/non-inferior on 3/3 seeds (92%/100%/100%); (3)
   macro-average +0.0256, comfortably non-negative; (4) no regression
   anywhere — **seed2 and seed3 have zero fair-comparison losses at all**,
   better than seed0. **New finding this replication surfaced**: the
   `acdc/rain` loss pattern that recurred identically across all 4 devices
   on seed0 does **not** reproduce on seed2 or seed3 — likely a
   seed0-specific calibration quirk, not a general router weakness (this
   narrows, but doesn't remove, the bounded-diagnostic task below).
   **Admissible claim**: *"Candidate-specific, device-conditioned routing
   consistently improves the risk–latency trade-off over single-probe
   rank-based routing across three independently trained supernets."*
   Still not "reliable" — `acdc/night`'s weak `probe_signal_aurc` (seed0:
   0.2616) not yet re-checked on seed2/seed3. B/C stayed seed0-only, as
   agreed (scoped mechanistic ablation).

   *(Original pre-registration, for reference:)* Do not use
   "reliable" while risk coverage under `acdc/night` remains weak.

   **Closing requirement 2 — real end-to-end router overhead on E1/E3.
   Protocol locked with Codex 2026-09-22 (revised after initial proposal):**

   - **Formula**: `t_e2e = t_probe + t_decision + t_switch + t_selected_extra`
     (`t_selected_extra = 0` when the chosen candidate is the probe level
     itself, reusing its output). **Measure `t_e2e` directly, as one
     end-to-end trace** — the 4-component decomposition is a sanity check
     only, never 4 independent benchmarks summed (synchronization/runtime
     overlap can make a naive sum wrong). CUDA calls must synchronize at
     timer boundaries; use one host monotonic clock for end-to-end on both
     backends; Hailo timing uses host wall-clock around request
     completion, never mixed with the runtime's internal profiler numbers.
   - **Frontier update — do NOT add the full measured total to the
     existing LUT** (the LUT already contains the selected candidate's own
     latency; adding total again double-counts it). Instead: **replay the
     already-locked held-out routing decisions**, map each image to a
     route class (`tiny→tiny`/`tiny→small`/`tiny→medium`/`tiny→large`),
     and substitute the corresponding directly-measured end-to-end latency
     for that route class. Recompute A and D's frontier, budget
     violations, and routing distribution from that replay.
   - **Measure both A and D**, not just D — A's decision path (entropy/
     calibrator + rank policy) vs. D's (per-candidate calibrators +
     constrained policy) may differ in cost; microbenchmark the decision
     component separately if useful, but the conclusion must come from an
     end-to-end replay of both policies, never an assumption they're equal.
   - **Warm/resident** (the headline deployment scenario): every needed
     engine/context already loaded and warm; `t_switch` is real dispatch/
     context-selection cost only. Must record peak memory and *confirm*
     concurrent residency is actually feasible on the device — if not
     enough memory to keep every engine resident, use a realistic cache
     policy instead (e.g. tiny always resident + one cached candidate),
     never claim warm-all-resident if that configuration can't actually be
     deployed.
   - **Cold/reload** (a stress-test upper bound, not the default
     deployment number unless a real app actually unloads engines every
     frame): candidate engine/context not resident, must genuinely
     release/reload on switch; report as its own distribution, never mixed
     into the warm/steady-state mean.
   - **Route classes, kept separate, never collapsed into one "larger"
     case**: `tiny→tiny` (reuse, no re-inference — also a sanity check
     that the implementation isn't accidentally running tiny twice),
     `tiny→small`, `tiny→medium`, `tiny→large`. Aggregate afterward using
     each policy's *real* routing distribution from the held-out replay.
   - **E1 + E3 are sufficient** for this closing requirement (2
     representative backends: Hailo/HEF vs. TensorRT/CUDA) — but the claim
     scope must say *"validated on two representative hardware
     backends,"* not extrapolate to E2/E5. Only extend to E2/E5 if:
     runtime/memory mechanism differs significantly, E1/E3 give
     conflicting conclusions, or overhead changes the GO conclusion and
     the scope needs pinning down.
   - **Repetition/thermal protocol, locked before running**: warm/resident
     — idle stabilization + record starting temperature; ≥50 warm-up
     iterations per route class; ≥500 measured iterations per route class;
     short blocks with interleaved/randomized route order (reduce thermal/
     order bias); if the median's bootstrap 95% CI is wider than 2%,
     increase up to 2000 iterations. Cold/reload — ≥30 iterations minimum,
     50 preferred, per transition, each a genuine release+reload; report
     its own distribution (load time is typically skewed). Report at
     minimum: median/mean/p95/p99; bootstrap 95% CI of median and mean;
     start/end temperature, clock/power mode, runtime version; iteration
     count, warm-up count, concurrent load, peak memory; throttling rate
     or excluded runs. **Never use `nvitop` as the primary latency
     source.** Lock the same power mode/clock policy across runs on Jetson
     where the infra allows, and record that config. If temperature
     crosses a throttling threshold or clock drops, stop, let it cool, and
     rerun that block — never silently pool throttled data.
   - **Final evaluation — report all 3 frontiers**: (1) the existing
     LUT-only frontier; (2) the warm end-to-end frontier (replay using
     warm/resident measurements); (3) the cold/reload stress-test
     frontier. **This closing requirement is satisfied when D still meets
     the GO criterion on the warm end-to-end frontier for both E1 and E3**
     — cold is a robustness/stress result, not automatically a fail
     condition unless it reflects the real deployment model.

   **E3 measured 2026-09-22, E1 measured 2026-09-28 — BOTH DONE, closing
   requirement 2 CONFIRMED on both backends.**
   `reports/router_overhead_v1_20260922.md`,
   `scripts/measure_router_overhead.py`, raw data
   `reports/router_overhead_E3_20260922.json`. Real infra work: E3 had no
   repo/pycuda before this — set up from scratch over SSH (installed
   `python3.8-dev`/`venv`, bootstrapped pip, built `pycuda==2022.2.2` from
   source since the latest PyPI release needs Python 3.10+, worked around
   TensorRT 8.5's removed `np.bool` alias, built ONNX exports + TensorRT
   engines directly on-device). Two real bugs caught before trusting any
   number (a flat-buffer reshape bug computing softmax over the entire
   ~1.4M-element array instead of the class axis; a non-contiguous `.T`
   transpose making subsequent numpy ops ~8x slower).

   **Load-bearing finding**: naive host-side (numpy/CPU) risk-score
   computation costs **~40-80ms per call** on E3's ARM/numpy build
   (confirmed via isolated microbenchmark — a real platform
   characteristic, not a bug) — large enough on its own to dominate and
   erase the router's entire hardware-aware-selection benefit if deployed
   this way (candidate-latency differences are only 0.94-9.41ms). A
   custom GPU-resident entropy kernel (matching the project's actual
   PyTorch `compute_risk_score`, which stays GPU-resident, never copying
   the full tensor to host) closes most of this gap: 2.5-21x speedup
   depending on route class. **All frontier analysis uses the GPU-kernel
   numbers**, not the naive-numpy ones.

   Real warm/resident e2e latency (GPU-kernel, median, n=500,
   all CI widths <0.5%): tiny→tiny 2.07ms, tiny→small 5.23ms,
   tiny→medium 11.97ms, tiny→large 26.70ms — all noticeably higher than
   the pure LUT-only candidate latencies (0.94/1.80/4.61/9.41ms), a real
   dispatch/copy overhead beyond pure inference. Cold/reload costs 2-3x
   the warm number. Decision-path cost (A: 0.053ms, D: 0.201ms) is
   negligible next to this. No throttling (32.0°C→38.5°C→37.5°C).

   **Initial frontier estimate (aggregated routing-distribution
   reweighting)**: showed D producing a wider, more useful quality-latency
   frontier than A. Codex reviewed this as **not sufficient to close the
   requirement** — the policy's own budget-feasibility decision must be
   re-run against the real latency table, not just relabeled after the
   fact.

   **Full per-image, e2e-aware replay: DONE 2026-09-22, CONFIRMED.**
   Built `scripts/evaluate_router.py --dump-per-image` (persists, per
   fit-half AND held-out image: raw probe score, each level's own
   ground-truth confusion matrix, fitted per-level calibrators — enough
   to replay any re-routing exactly, no re-inference) and `scripts/
   replay_router_with_overhead.py` (two strictly separate modes:
   **post-hoc** — original LUT-based decision, scored with real e2e
   latency, confirms the old latency model was badly wrong in absolute
   terms; **e2e-aware** — decision *recomputed* using real e2e latency as
   both the ranking/budget input and the new device-budget grid
   `[2.07, 5.23, 11.97, 26.70]` ms, operating point re-selected via
   `select_budget_matched_operating_point` on fit-half stats only, never
   touching held-out first — the real deployment result).

   | | value |
   |---|---|
   | Total cells (5 splits × 4 e2e budgets) | 20 |
   | Fair cells (A itself doesn't violate) | 12/20 |
   | D vs. A on fair cells | **8 wins, 3 ties, 1 loss** (67%) |
   | Mean (D − A), fair | **+0.0207** |
   | Mean (D − oracle), fair | −0.0057 |
   | **D violations, all 20 cells** | **0/20** |
   | A violations, all 20 cells | 8/20 (mean rate 8.95%) |

   Confirms the earlier approximation almost exactly. `reports/
   router_overhead_replay_E3_20260922.json` (raw), `reports/
   router_overhead_v1_20260922.md`'s 2026-09-22 update (full writeup).

   **E1 (Hailo-8) measured for real 2026-09-28** once the device came back
   online (confirmed a real connectivity/power issue, not an account
   problem, as suspected). Direct-SSH setup, same discipline as E3:
   recompiled all 4 HEFs fresh from the current post-rescale ONNX exports
   (the week-1-2 smoke-test HEFs were the old ~5K-param placeholder
   architecture and no longer resident on-device after reboot) via the
   cached Hailo DFC 3.34.0 toolchain; new harness `scripts/
   measure_router_overhead_hailo.py` against the real `hailo_platform`
   (pyhailort) API. **A second, different load-bearing platform finding**:
   unlike E3 (naive numpy entropy dominates), this Hailo-8 module allows
   only ONE network group hardware-activated at a time, so even the
   warm/resident scenario pays a real activate/deactivate cycle on every
   inference — that cycle, not entropy computation, dominates E1's
   overhead. No GPU-resident-kernel fix is possible here by hardware
   necessity (Hailo's dataflow architecture has no CUDA-like
   compute-shader model) — only a numpy/host entropy backend exists on
   this device. Two further real infra findings while building the
   cold/reload scenario: `ConfiguredNetwork` has no release/deconfigure
   method at all (a `VDevice` accumulates configured groups for its whole
   lifetime, capped at 32 "core-ops" — the first cold-scenario design hit
   this cap after ~28 reloads); and this chip allows exactly one `VDevice`
   at a time (a concurrent second one raises
   `HAILO_OUT_OF_PHYSICAL_DEVICES`) — so E1's cold scenario necessarily
   reconfigures the *entire* VDevice (probe included) every iteration,
   stricter than E3's candidate-only reload; disclosed explicitly as a
   hardware-forced design difference, not a shortcut.

   Real warm/resident e2e latency (numpy, only backend, median, n=500, all
   CI widths ≤1.2%): tiny→tiny 34.95ms, tiny→small 46.13ms, tiny→medium
   63.36ms, tiny→large 92.68ms — all far above the LUT-only streaming
   numbers (3.72/6.63/20.47/41.18ms), consistent with the
   activate/deactivate-dominated mechanism above. Cold (full reconfigure)
   120.0/156.0/176.0/214.6ms. Decision-path cost (A: 0.012ms, D: 0.045ms)
   matches E3's order of magnitude, as expected (hardware-agnostic).
   No throttling (SoC 49.6→55.1→54.6°C; Hailo-8 chip telemetry
   46.1→48.5→47.7°C).

   **Full per-image, e2e-aware replay on E1, same locked methodology**:
   `reports/router_overhead_replay_E1_20260928.json`.

   | | value |
   |---|---|
   | Total cells | 20 |
   | Fair cells | 12/20 |
   | D vs. A on fair cells | **8 wins, 3 ties, 1 loss** (67%) |
   | Mean (D − A), fair | **+0.0207** |
   | Mean (D − oracle), all | −0.0036 |
   | **D violations, all 20 cells** | **0/20** |
   | A violations, all 20 cells | 8/20 (mean rate among violators 22.36%) |

   **Essentially an exact qualitative and near-exact quantitative
   replication of E3's result**, despite the two backends' overhead being
   dominated by completely different mechanisms and E1's absolute
   latencies running 2-4x higher than E3's. Full writeup: `reports/
   router_overhead_v1_20260922.md`'s 2026-09-28 update.

   **Closing requirement 2 is now CONFIRMED on both backends.** Admissible
   claim, extended: *"Candidate-specific, device-conditioned routing (D)
   continues to win under real, directly measured end-to-end overhead —
   including router-specific costs invisible to a pure inference-latency
   lookup table — on two structurally different accelerator backends
   (TensorRT/CUDA GPU and Hailo-8 dataflow NPU), despite those two
   backends' overhead being dominated by different mechanisms."*

   **GPU-kernel correctness audit vs. PyTorch reference: DONE 2026-09-28,
   PASS — the mandatory pre-manuscript-lock requirement.** Codex's explicit
   instruction: audit, not a new experiment — feed the CUDA kernel and the
   numpy transcription of `router.risk_probe.compute_risk_score`'s exact
   formula the SAME synthetic logits, at E3's real (on-disk `.engine`
   file) output shapes for all 4 levels. `scripts/audit_gpu_risk_kernel.py`,
   raw `reports/audit_gpu_risk_kernel_E3.json`.

   | criterion | tolerance (locked before running) | observed (max) | pass? |
   |---|---|---|---|
   | per-pixel max abs error | ≤1e-3 nats | 9.54e-07 | **PASS** |
   | scalar risk-score abs error | ≤1e-4 nats | 2.38e-07 | **PASS** |
   | decision_a agreement (2000 trials) | ≥99.9% | **100.000%** | **PASS** |
   | decision_d agreement (2000 trials) | ≥99.9% | **100.000%** | **PASS** |

   Zero decision mismatches across 2000 trials at every level — the two
   implementations run the identical formula, so the only possible source
   of disagreement is FP32 summation-order non-associativity, which stays
   ~1000x under tolerance at this array size. Tie-handling is moot: no tie
   case occurred to characterize. **This closes the last mandatory
   pre-manuscript-lock item from this review.**

   **Cross-seed E2E replay: DONE 2026-09-29, CONFIRMED across all 3
   seeds and both backends (canonical-grid corrected 2026-09-29).**
   Codex's instruction: the 20-cell replay above used seed0 only, while
   the router method itself was already 3-seed confirmed (closing
   requirement 1) — replay (no new hardware measurement) the existing
   E1/E3 overhead JSONs against fresh per-image dumps from
   `pace_seg_v1_seed2`/`pace_seg_v1_aug_seed3`.

   **A real bug, caught by Codex's review**: the first pass used
   seed0-era `router_{E1,E3}_20260921.json` as `--original-results-json`
   for *every* seed, so seed2/seed3 replays used **seed0's**
   `risk_target_grid` by mistake. Fixed per Codex's protocol: lock ONE
   canonical grid per seed (`reports/router_E3_seed{2,3}_dump_meta.json`,
   generated in the same invocation as that seed's per-image dump) and use
   it for **both** E1 and E3 (only the E2E latency table changes per
   backend). Seed0 was already canonical by coincidence — `router_E1_
   20260921.json` and `router_E3_20260921.json`'s grids are byte-identical
   — so seed0's numbers are unchanged.

   **Canonical per-seed table:**

   | device | seed | fair | W/T/L | win+tie | mean(D−A) | D viol | A viol |
   |---|---|---|---|---|---|---|---|
   | E3 | 0 | 12/20 | 8/3/1 | 91.7% | +0.0207 | 0/20 | 8/20 |
   | E3 | 2 | 10/20 | 7/3/0 | 100.0% | +0.0241 | 0/20 | 10/20 |
   | E3 | 3 | 13/20 | 11/2/0 | 100.0% | +0.0273 | 0/20 | 7/20 |
   | E1 | 0 | 12/20 | 8/3/1 | 91.7% | +0.0207 | 0/20 | 8/20 |
   | E1 | 2 | 10/20 | 7/3/0 | 100.0% | +0.0241 | 0/20 | 10/20 |
   | E1 | 3 | 12/20 | 10/2/0 | 100.0% | +0.0276 | 0/20 | 8/20 |

   **Aggregated (Codex's requested summary format)**:

   | Backend | Fair | W/T/L | Win-or-tie | Macro ΔmIoU | D viol. | A viol. |
   |---|---|---|---|---|---|---|
   | E3 | 35/60 | 26/8/1 | 97.1% | +0.0241 | 0/60 | 25/60 |
   | E1 | 34/60 | 25/8/1 | 97.1% | +0.0241 | 0/60 | 26/60 |
   | **Overall** | **69/120** | **51/16/2** | **97.1%** | **+0.0241** | **0/120** | **51/120** |

   (+0.0241 mIoU = **+2.41 mIoU points**.) **69/120 counts operating
   cells, not independent statistical samples** — E1/E3 share the same
   per-image predictions from the same 3 checkpoints; this is
   cross-backend deployment validation, not independent accuracy
   replication. Seed2's identical stats between E3/E1 illustrate why:
   both A and D reduce to rank-among-4-device-latencies comparisons when
   the budget grid is built from those same latencies, so identical
   decisions across backends are the *expected* outcome when latency
   rankings agree — **not** evidence D independently "chooses a different
   architecture per device." Call the method **hardware-cost-conditioned**
   (or device-cost-conditioned) routing, not "device-conditioned
   decisions." Seed3 shows this isn't purely mechanical: E3/E1 diverge by
   one fair cell there, traced to `select_budget_matched_operating_point`'s
   continuous fit-half mean-latency landing on different sides of a
   budget threshold per device — legitimate, not a bug.

   **Updated admissible claim (Codex's template, canonical numbers
   substituted — Codex's own draft table was explicitly provisional
   pending this fix)**:
   *"Using directly measured end-to-end route costs, candidate-specific,
   hardware-cost-conditioned routing outperformed or matched the
   rank-based policy in 67 of 69 fair operating cells (51 wins, 16 ties, 2
   losses) across three independently trained supernets and two
   structurally different accelerator backends. It achieved a
   macro-averaged gain of 0.0241 mIoU (2.41 points) and produced no
   budget-violating operating point across all 120 evaluated cells."*
   Boundary (state alongside the claim, every time): *"The two backend
   evaluations share model predictions and therefore demonstrate
   cross-backend deployment robustness rather than independent accuracy
   replication."*

   Raw: `reports/router_overhead_replay_{E1,E3}_seed{2,3}_20260929.json`
   (corrected in place). Full writeup: `reports/router_overhead_v1_
   20260922.md`'s 2026-09-29 update.

   **CUDA-kernel-vs-production-PyTorch audit on real engine outputs:
   DONE 2026-09-29, PASS.** Codex's second correction: the first audit
   compared against a numpy transcription on synthetic logits — good
   arithmetic evidence but not a true "PyTorch-reference" audit.
   `scripts/audit_gpu_risk_kernel_production.py` calls the real
   `router.risk_probe.compute_risk_score` directly, against **40 real
   TensorRT outputs (10 samples × 4 levels)** on E3, via the actual
   deployed `infer_no_copy` + `risk_score_gpu` path. Per Codex's framing:
   the earlier 2,000-*synthetic*-trial audit is an arithmetic stress test
   (broad numerical coverage); these 40 *real* outputs are a narrower
   integration audit (real buffer layout/dtype/sync/indexing). Scalar
   risk-score abs error: 7.15e-07 max over the 40 (tolerance 1e-4, ~140x
   margin). Decision agreement (real `RiskCalibrator`/`_select_by_risk`/
   `_select_by_risk_and_latency_budget`, checked on the 10 real
   tiny-level/probe samples): **100.000% (0/10 mismatches)**. 99.9% is
   the pre-registered acceptance threshold, not a statistical estimate
   from N=10/40 — too small to resolve a rate that precisely; 0/10 and
   0/40 are the exact evidentiary counts. Raw:
   `reports/audit_gpu_risk_kernel_production_E3.json`.

   **Both Codex-mandated correctness items from the 2026-09-29 review
   are closed. The evidence package for closing requirement 2 is now
   complete and canonical.**

   **FINAL LOCK (Codex, 2026-09-29): closing requirement 2 is PASS,
   permanently locked.** No correctness or bookkeeping item remains
   open; temporal-window routing, UIoU, cold-reload replay, and the
   numpy-overhead decomposition are confirmed **not required** to
   complete this paper. **Three annotations that must accompany the
   claim in the manuscript, every time it is stated** (full text and
   rationale in `reports/router_overhead_v1_20260922.md`'s FINAL LOCK
   section): (1) the 120 (or 69) operating cells are not independent
   statistical samples — never use them as n for a significance test or
   CI; (2) W/T/L and win-or-tie use the 69-fair-cell denominator, budget
   violations use the full 120-cell denominator — never blend the two;
   (3) zero violations follows partly from D's own hard-budget-by-design
   policy — the finding is that D keeps its quality advantage *while*
   satisfying that constraint, not "zero violations" alone.

   **Artifact freeze**: after this point, do not modify the policy code,
   risk grid, data split, or metric definitions this evidence package
   depends on — a manuscript-wording-only change never requires
   reopening it; a change to any of those does. Full freeze manifest
   (commit reference, canonical result/audit files, seed/checkpoint IDs,
   per-seed risk-grid source, E1/E3 latency-table IDs, exact replay/audit
   commands, final locked table) recorded in `reports/
   router_overhead_v1_20260922.md`'s "FINAL LOCK, 2026-09-29" section —
   not duplicated here to avoid two sources of truth drifting apart.

   **Next step per Codex's locked order: move to manuscript (Related
   Work/Introduction/Discussion/Limitations), then lock Abstract/Results
   tables/Conclusion. Temporal-window routing and UIoU move to Future
   Work — not opened in this paper's scope.**

   **`acdc/rain` loss pattern**: investigate as a **bounded diagnostic
   only** — do not tune the policy on the held-out rain split. Extract:
   (D−A) mIoU *and* latency delta (a small accuracy loss for a large
   latency saving would be a Pareto trade-off, not a failure); routing
   distribution by level for A and D; per-level, per-quantile calibration
   residuals; oracle regret and the rate at which the true best choice is
   non-monotonic in model size. Since the pattern is identical across all
   4 devices, the cause is likely in the risk signal/calibration or the
   rain condition itself, not the hardware LUT. If the diagnostic suggests
   a fix, pre-register it and test on a fresh validation split — never
   fix-then-report on the same held-out split. If not, report as a
   disclosed limitation. **`probe_signal_aurc`=0.2616 on `acdc/night` must
   be stated plainly as a limitation** — more important than the average
   inversion rate: the cheapest probe loses risk-ranking ability exactly
   in the hardest domain.

   **Temporal-window routing / UIoU**: not opened as full branches yet.
   Temporal-window routing only starts once overhead shows engine-
   switching/probe cost is non-trivial AND real frame-ordered data exists
   (no synthetic frame-order streaming result). UIoU: feasibility-audit
   ACDC's annotations first; stop if no genuine uncertainty-region label
   exists — never synthesize pseudo-labels to use as headline evidence.
   Neither is prioritized over the two closing requirements above.

   **Not yet**: closing requirement 2 (real E1/E3 router-overhead
   measurement, mandatory) — closing requirement 1 is done; bounded
   `acdc/rain` diagnostic (lower priority now that it's confirmed
   seed0-specific); `acdc/night`'s weak `probe_signal_aurc` not yet
   re-checked on seed2/seed3; temporal-window routing and UIoU
   (conditionally unblocked, not started, gated behind overhead
   measurement).
4. ~~**RQ1 budget sweep**~~ **Done 2026-09-20** (`reports/rq1_budget_sweep_v1_20260920.md`,
   `src/imavis_edge_seg/search/flops.py::evaluate_flops_proxy_at_budget`,
   `scripts/rq1_budget_sweep.py`). 640 evaluations (40 budgets x 4 reference x 4
   target devices). Headline: even self-calibration mis-selects 37.5-65% of the
   time; cross-device transfer from the 2 fastest devices to the 2 slowest is
   often a real budget *violation* (mean slack -3.9 to -12.1ms), not just
   accuracy loss. README Phase 5 and `docs/MANUSCRIPT_DRAFT.md` §3.2 updated.
5. **Router overhead measurement**, **temporal-window routing**, **UIoU** — not
   started, see `reports/router_v1_20260914.md`'s "Not yet done".

## How to claim an experiment_id

Add a row to the table above (experiment_id, owner, one-line purpose, status:
claimed/running/done) *before* running `start_train_supernet.sh`/
`start_train_baseline.sh`. Remove the row (or mark done + link the report) once
evaluated and written up.
