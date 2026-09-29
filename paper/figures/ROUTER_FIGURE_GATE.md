# Figure 6 evidence gate: router quality versus real route cost

Figure 6 is not manuscript-final in this snapshot. The current router fitting path
computes probe entropy with a ground-truth ignore mask on the fit half, while deployment
and held-out evaluation compute the feature without ground truth (`scripts/evaluate_router.py`,
`_collect_split_data`). This is a train/deployment feature mismatch rather than held-out
label leakage, but it prevents freezing the final calibrated-router headline.

Before generating Figure 6:

1. compute the probe feature identically during calibrator fitting and deployment;
2. rerun the locked three-run, two-backend replay with one canonical risk grid per
   training run;
3. preserve fit-half selection and held-out-only reporting;
4. regenerate quality versus directly measured end-to-end route cost from canonical
   machine-readable artifacts;
5. report operating cells as correlated operating points, not independent samples.

The plotting script should be added only after the corrected artifacts exist. It must
parse those artifacts directly and must not copy the previous 120-cell numbers into
source code.
