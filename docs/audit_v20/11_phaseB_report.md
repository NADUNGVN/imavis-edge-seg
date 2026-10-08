# 11 — Phase B robustness / sensitivity analyses (no new training, no device work)

All analyses replay stored per-image predictions and measured cost tables. They are reported as robustness checks; the primary results in the manuscript are unchanged.

## B1. Per training seed (`scripts/phaseB_per_seed.py` → `reports/phaseB_per_seed_20261004.json`)
Same operating-point selection as the pooled analysis; each run summarized separately (no bootstrap).

| Comparison (median costs) | Run A (seed 0) | Run B (seed 1) | Run C (seed 2) | Pooled (Table 7) |
|---|---|---|---|---|
| D − A, fair cells | +2.70 (21; 17/2/2) | +3.13 (23; 21/2/0) | +2.74 (18; 16/2/0) | +2.87 (62) |
| D − A-hard | +1.33 (40; 16/24/0) | +1.56 (40; 16/24/0) | +1.35 (40; 16/24/0) | +1.41 (120) |
| D − T-hard | −0.11 (40; 4/28/8) | −0.06 (40; 4/28/8) | +0.09 (40; 8/24/8) | −0.03 (120) |
| D − static, route cost | −0.19 | −0.18 | −0.13 | −0.16 |
| D − static own cost, AGX, feasible | −3.05 (35) | −3.00 (35) | −2.94 (35) | −3.00 (105) |
| D − static own cost, Hailo-8, feasible | −8.42 (25) | −8.18 (25) | −7.95 (25) | −8.18 (75) |

Every qualitative conclusion holds in every seed. Range across seeds: D − A 2.70–3.13; D − A-hard 1.33–1.56; D − T-hard −0.11 to +0.09 (sign changes across seeds → consistent with no detectable difference); static advantage 2.94–3.05 (AGX), 7.95–8.42 (Hailo-8). With three seeds this range is descriptive, not a seed-variance estimate.

## B2. Sequence-aware split (`scripts/phaseB_sequence_split.py` → `reports/phaseB_sequence_split_20261004.json`)
**Feasibility check.** Sequence identifiers exist in the manifests: Cityscapes `city_sequence` (235 groups; frankfurt_000001 alone has 217 images), ACDC recording IDs (fog 3, night 2, rain 8, snow 8). Per-image predictions are stored for all validation images (fit + held-out halves re-assembled in manifest order; length checks pass).

**Procedure.** Whole groups assigned greedily (largest first) to the smaller half; per-condition and pooled calibrators, risk-target grid, T-hard thresholds and operating points refitted on the new fit half only; held-out labels used only for scoring.

**Limitation (not hidden).** Night (2 recordings) and fog (3) give one or two groups per half, so the held-out half of these conditions is a different recording — a cross-sequence test rather than a random split; Cityscapes is dominated by one large sequence. The split is coarse.

| Result (median) | Alternating index (V21) | Sequence-grouped |
|---|---|---|
| Split sizes (fit/held-out) | 250/250, 50/50, 53/53, 50/50, 50/50 | 250/250, 46/54, 56/50, 51/49, 51/49 |
| A violating cells / 120 | 58 | 65 |
| D violating cells / 120 (route grid) | 0 | 0 |
| D − A, fair | +2.87 (62) | +3.38 (55; 49/6/0) |
| D − A-hard | +1.41 | +1.71 (50/70/0) |
| D − T-hard | −0.03 | −0.04 (8/80/32) |
| D − static, route cost | −0.16 | −0.20 |
| D − static own cost, AGX, feasible | −3.00 (105) | −2.98 (105; 0/42/63) |
| D − static own cost, Hailo-8, feasible | −8.18 (75) | −8.39 (75; 0/4/71) |

All main conclusions are unchanged under sequence-grouped splitting. Calibration transfer across recordings did not erase the router ablation result.

## B3. Break-even uncertainty (`scripts/phaseB_breakeven_bootstrap.py` → `reports/phaseB_breakeven_bootstrap_20261004.json`)
Image-level bootstrap (200 replicates, seed 0) of the counterfactual mean-cost break-even overhead; choices depend only on probe scores and calibrators, so only scoring is resampled. Linear interpolation on the α grid (0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0) adds discretization error that the interval does not capture.

| Device | Break-even (point) | 95% bootstrap interval | Measured overhead |
|---|---|---|---|
| AGX Xavier | 1.27 ms | [0.57, 1.50] ms | 1.9 ms |
| Hailo-8 | 4.26 ms | [2.13, 5.15] ms | 44 ms |

All 200 replicates crossed zero on both devices; zero-overhead gain 95% interval: [0.65, 1.58] (AGX) and [0.69, 1.61] (Hailo-8) points. Not included: training-seed variability, timing-measurement variability.
