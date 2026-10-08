# 04 — Results consistency audit

## R-1 (P0) Static vs adaptive comparison includes cells with no feasible route — VERIFIED ERROR (MAJOR)
**Location:** Table 7 rows "D − best feasible static, own cost, AGX Xavier / Hailo-8"; Table 7 caption ("A-hard, T-hard, D, pooled D, static, and LOCO D 0 of 120 in every table"); §5.3 "Against static deployment"; Fig. 7 last two rows.

**Evidence (`reports/audit_static_cells_20261004.json`, produced by `scripts/audit_static_cells.py`, read-only):**
- Own-cost comparisons use the union of the route and static costs of one device as budgets: 8 budgets × 3 runs × 5 splits = 120 cells *per device*. The caption instead describes 120 as runs × two backends × splits × four budgets, which is the router-comparison grid. Denominator description is wrong for these rows.
- AGX Xavier: budget 1.18 ms is below the cheapest route (1.84 ms): **15 cells** with no feasible route. Hailo-8: budgets 6.27, 12.07, 30.91 ms are below 44.26 ms: **45 cells**.
- In those cells `d_choice` falls back to tiny and **violates the budget** (D_violating_cells = 15 and 45). `router_ondevice_replay_20261004.json` also records D violating cells (5 on AGX, 15 on Hailo-8 for Run A). The caption claim "D 0 of 120 in every table" is therefore false for the own-cost rows.
- Restricting to cells with at least one feasible route: AGX Xavier −3.00 points (0/46/59 over 105 cells); Hailo-8 −8.18 points (0/6/69 over 75 cells). Reported (all cells): −2.62 and −7.41.

**Impact:** The negative result is *preserved and slightly stronger*; the issue is reporting correctness (denominator, violation claim). Text already says "at the lowest budgets … no route is feasible", but the counts mix those cells in as ties/losses.

**Proposed fix (no new experiment):** report own-cost rows over feasible cells (105 / 75) with violations stated, or report both; correct caption; recompute bootstrap CI on the feasible subset (cheap re-run of existing script).

## R-2 (P0) §5.2 "1.5-point band" sentence — SUSPECTED ISSUE (wording), numbers correct
The band sentence refers to "an independently trained model of similar size", i.e. Fast-SCNN. Against Fast-SCNN the differences are +0.75, −1.80, −0.77, −3.95, −1.38 → within 1.5 on Cityscapes, night, snow — the sentence is numerically correct. Against PACE-Large (the RQ2 focus) all ACDC gaps exceed 1.5 (−1.74, −2.75, −1.61, −1.71). Because the paragraph and Fig. 6 now emphasize shared-vs-alone, a reader will read the band against PACE-Large and see a contradiction.
**Fix:** name Fast-SCNN explicitly in that sentence and add the PACE-Large statement ("against the same architecture trained alone, the gap exceeds the band on all four ACDC conditions").

## R-3 (P0) Deadline terminology — SUSPECTED ISSUE (MINOR)
§4.4 defines violations against tabulated median/p95/p99 costs (correct). Remaining stronger phrasing: Conclusion "its selected routes always fit the tabulated … costs" (OK); Abstract "routing under a hard budget" (OK). After R-1, add that in budgets below the cheapest route no policy that must run the probe can meet the budget. No "every-frame guarantee" wording found.

## R-4 Table 7 vs Fig. 7 — consistent values; layout mixes two question types (see 06).

## R-5 Abstract "Every candidate, every complete route, and every static deployment is timed in the same harness on edge devices" — VERIFIED ERROR (MINOR)
Candidate-only latency on four devices uses trtexec/hailortcli, not the same harness; the same harness covers only AGX Xavier and Hailo-8. Fix wording.

## R-6 "candidate-specific calibration adds nothing" (Abstract) — SUSPECTED ISSUE (MINOR, rule 6)
CI [−0.11, +0.09] contains zero; this supports "no detectable gain; gains above 0.09 points are excluded at 95%", not equivalence. Replace "adds nothing"/"matches" with that bounded statement.

## R-7 Break-even is counterfactual — SUGGESTED IMPROVEMENT
1.3/4.3 ms come from scaling measured overhead by α with fixed predictions and decisions recomputed; no device was run at reduced overhead. State "in counterfactual replay" in Abstract/Fig. 9 caption.

## R-8 Hailo-8 overhead attribution — MISSING EVIDENCE
`measure_static_vs_route_hailo.py` (explicit path) activates each network group once for static runs but activates/deactivates per call for routes; only totals are timed. Manuscript §5.1 says "network-group activation and host-side entropy dominate". This is plausible but not measured. Fix: rephrase as "the route includes per-call activation of two network groups and host-side entropy; the components were not timed separately", or run a component timing (requires device time and approval).

## R-9 Compiled validation scope — verified
Compiled accuracy and on-device replay: Run A only, all five splits, held-out halves, both devices. Manuscript Limitations states "one of the three supernets" (correct).

## Other checks (passed)
RQ1 numbers, RQ2 means/s.d., all Table 7 router rows, Table 8, calibration table, compiled table, Table 6 latencies: match artifacts to reported precision.
