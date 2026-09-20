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
  This is not a formality — three real bugs already happened this way this session
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
  job ran or never started is still being diagnosed separately, see below).

## Active / claimed experiment_ids

| experiment_id | owner | purpose | status |
|---|---|---|---|
| `pace_seg_v1_qat_calibrated_ema_percentile_seed0` | Claude | QAT 2x2 screen cell 2: shared supernet x EMA/percentile observer, `large`, seed0 | **done** — worst-case −0.97, see below |
| `qat_exported_large_dynamic_seed0` | Claude | QAT 2x2 screen cell 3: exported-subnet x dynamic range, `large`, seed0 | **done** (first attempt failed with CUDA OOM on SERVER-03, no checkpoint written; retried successfully) — worst-case −1.55, see below |
| `qat_exported_large_ema_percentile_seed0` | Claude | QAT 2x2 screen cell 4: exported-subnet x EMA/percentile observer, `large`, seed0 | **done** — worst-case −0.70 (best cell), see below |
| `qat_exported_large_ema_percentile_seed3` | Claude | QAT 2x2 screen: seed3 confirmation of the best cell (4) | **done — CONFIRMED**, worst-case −0.51, mean −0.32, see below |
| `pace_seg_v1_qat_calibrated_ema_percentile_seed3` | Claude | Mechanistic replication: shared supernet x ema_percentile, seed3 — does the fix rescue the shared model too? | **done — yes, also passes**, worst-case −0.86, mean −0.50, see below |
| `pace_seg_v1_qat_calibrated_ema_percentile_seed2` | Claude | Codex's closing step 1: shared x ema_percentile on `pace_seg_v1_seed2`, the seed dynamic-QAT failed worst on (−2.54) — paired 3/3-seed confirmation attempt | claimed, not yet launched |

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
   1. Run cell 2 (shared × ema_percentile) on `pace_seg_v1_seed2` — the
      dynamic-QAT seed that failed worst (−2.54). If worst-case ≤1.5, that's
      a paired 3/3-seed confirmation; if not, the claim stays at 2/3 seeds.
      **Launched, training in progress on SERVER-03, not yet evaluated.**
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

   After these two steps: **freeze QAT regardless of outcome**, no further
   observer/hyperparameter search, shift effort to the router (open thread
   #2/#3). QAT remains a **secondary contribution**; no INT8 headline claim
   before real compiled-engine (TensorRT/Hailo) accuracy+latency numbers
   exist (everything so far is PyTorch fake-quantization simulation). The
   outlier-sensitivity explanation remains consistent with the data, not
   directly demonstrated (would need activation max/percentile diagnostics)
   — noted as future work, not blocking the freeze.
2. **Router candidate-specific per-level risk/error model.** Must be fit on the
   **fit-half only** (the half `evaluate_router.py` already reserves for
   calibrator fitting) — never on the test-half, even though the test-half
   already computes every level's prediction per image (that's for *scoring*
   policies, not for *fitting* one; using it to fit would leak).
3. **Router latency-value-aware policy.** `_select_by_risk` (`router/policy.py`)
   currently uses only each candidate's latency *rank* after sorting, never the
   magnitude. Data to fix this already exists (`outputs/benchmark_lookup_table.csv`,
   4 real devices). No new server run needed for the policy change itself.
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
