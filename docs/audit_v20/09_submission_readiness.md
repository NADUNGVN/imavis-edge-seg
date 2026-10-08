# 09 — Submission readiness

**Verdict: not ready.** One administrative blocker (I01) and one reporting error (I02) must be fixed; the scientific conclusions hold and are not weakened by any finding.

## Top 10 risks to acceptance
1. Placeholders in author/affiliation/funding/repository (I01) — desk rejection.
2. Static-comparison denominator and false "0 violations" claim for own-cost rows (I02).
3. Reviewer objection: uncertainty ignores seed variance (I03).
4. Reviewer objection: correlated alternating-index split (I04).
5. Narrow scope: four candidates, one probe, two datasets → "is the negative result general?".
6. Hailo-8 overhead looks implementation-specific (per-call activation); unmeasured components (I08).
7. Equivalence wording for D vs T-hard (I07).
8. RQ2 band sentence read as contradiction (I05).
9. Fig. 10 does not show the Small prediction (I09).
10. Bibliography not fully verified (I13).

## Corrections possible without new experiments
I02 (re-run existing analysis on feasible cells), I03 (per-run deltas from existing dumps), I05, I06, I07, I09, I10, I11, I12, I13, I16.

## Minimal analyses/experiments that would materially raise confidence
- (No device time) Sequence-grouped fit/held-out re-split and replay (I04).
- (No device time) Bootstrap CI for break-even overheads.
- (Device time, needs approval) Hailo-8 per-component route timing: probe inference, host entropy, activation/deactivation, transfer, selected inference.
- (Device time, needs approval) Compiled-engine accuracy for Runs B and C.

## Optional, lower priority
Class-balanced risk target and alternative entropy statistics (S-7); AGX bar in Fig. 1(c).

## Revised figure plan
Fig. 7 → two panels (ablation | static over feasible cells); Fig. 10 → add Small column + route-frequency inset; Fig. 5/9 axis labels; Fig. 3 caption "schematic". Originals kept in `paper/figures/generated/` for side-by-side comparison.

## Submission checklist
- [ ] Author metadata, e-mail, affiliation (I01)
- [ ] Funding, acknowledgements, repository/DOI (I01)
- [ ] AI-use declaration lists every tool used (I14)
- [ ] Table 7 / Fig. 7 static rows corrected (I02)
- [ ] Wording fixes I05–I07, I11
- [ ] Figure fixes I09, I10, I12, I16
- [ ] Bibliography normalized and verified (I13)
- [ ] Supplement consistent with revised Table 7
- [ ] Final PDF visually inspected page by page (Overleaf build)
- [ ] Journal scope/format confirmed with author (IVC provisional)

Nothing in this checklist has been marked complete; no manuscript edits were made during the audit.
