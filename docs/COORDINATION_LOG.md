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
  This is not a formality — two real bugs already happened this way this session
  (a baseline checkpoint silently overwritten by a same-named concurrent run; a
  status script reading the wrong server's job through a shared pointer file,
  since fixed in `scripts/server/status_train_supernet.sh`).

## Active / claimed experiment_ids

| experiment_id | owner | purpose | status |
|---|---|---|---|
| `pace_seg_v1_qat_calibrated_ema_percentile_seed0` | Claude | QAT 2x2 screen cell 2: shared supernet x EMA/percentile observer, `large`, seed0 | claimed, not yet launched |
| `qat_exported_large_dynamic_seed0` | Claude | QAT 2x2 screen cell 3: exported-subnet x dynamic range, `large`, seed0 | claimed, not yet launched |
| `qat_exported_large_ema_percentile_seed0` | Claude | QAT 2x2 screen cell 4: exported-subnet x EMA/percentile observer, `large`, seed0 | claimed, not yet launched |

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

1. **QAT rescue — 2×2 factorial screening, seed0 first, replicate winner on a
   2nd seed.** Cells: (shared weights × dynamic range) — **already done**, 3
   seeds, worst case −2.54; (shared × per-level EMA/percentile observer),
   (exported-subnet × dynamic), (exported-subnet × EMA/percentile) — **infra
   built and unit-tested 2026-09-20** (`training/quantization.py`'s new
   `ema_percentile` observer; `training/baseline_trainer.py`'s new `model=`/
   `calibration_level=` params; `training/data.py`'s new `level=` param;
   `scripts/train_exported_subnet.py`; `scripts/server/{start,run,status}_
   train_exported_subnet.sh`), **not yet run on a server** — 3 commands ready,
   see `reports/qat_rescue_2x2_screen_infra_20260920.md`. Deliberately *not*
   combining both axes in a single first attempt (would not distinguish which
   factor helped) — do the full 4-cell screen, then confirm only the winning
   cell on a second seed.
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
