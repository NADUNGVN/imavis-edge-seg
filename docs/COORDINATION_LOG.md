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

   **Closing requirement 1 — 3-seed replication (mandatory, router is
   flagship and carries a headline accuracy claim, so the project's
   already-locked 3-seed rule applies)**: rerun A/B/C/D on the 2 remaining
   supernet seeds (`pace_seg_v1_seed2`, `pace_seg_v1_aug_seed3`) if
   evaluation-only cost allows; at minimum, replicate A and D (B/C can
   stay seed0-only as a scoped mechanistic ablation, explicitly labeled as
   such). Locked before running: same split/features/calibrator/quantile
   grid/LUT/metric as seed0; each seed's calibrator fit only on that
   seed's own fit-half; **no policy changes after seeing seed2/seed3
   results**; report every seed individually AND the macro-average across
   3 seeds (not just a pooled cell count); report a bootstrap CI over
   images or a paired CI for (D−A), but do not treat correlated grid cells
   as independent samples. **Confirmation criteria**: (1) D still zero
   hard-budget violations; (2) D wins or is Pareto-non-inferior to A on
   ≥2/3 seeds; (3) the 3-seed macro-average (D−A) is non-negative at the
   locked comparison points; (4) no single seed shows a large, systematic
   regression across most conditions/devices. If met, the admissible claim
   is: *"Candidate-specific, device-conditioned routing consistently
   improves the risk–latency trade-off over single-probe rank-based
   routing across three independently trained supernets."* Do not use
   "reliable" while risk coverage under `acdc/night` remains weak.

   **Closing requirement 2 — real end-to-end router overhead on E1/E3
   (mandatory, high priority — current latency numbers are candidate-only
   from the LUT, not the router's own cost)**: measure separately probe
   pass, policy/calibrator computation, candidate inference, and
   engine-switch cost; report total latency both when tiny is selected
   and when a larger level is selected; cover warm steady-state and
   cold/switch-heavy cases; account for reusing the probe's own output
   when the probe level IS the selected candidate. Report
   `t_total = t_probe + t_decision + t_selected + t_switch` and the
   quality-latency frontier *after* adding overhead — D's advantage only
   "closes" the router contribution if it survives this addition.

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

   **Not yet**: seed2/seed3 replication (can run in parallel with
   overhead measurement); real E1/E3 router-overhead measurement;
   bounded `acdc/rain` diagnostic; temporal-window routing and UIoU
   (conditionally unblocked, not started).
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
