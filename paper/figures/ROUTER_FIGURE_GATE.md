# Figure 6 evidence gate: router quality versus real route cost

Figure 6 is not manuscript-final in this snapshot. The evaluator now computes the
same all-pixel entropy feature during calibrator fitting, held-out evaluation, and
deployment. Historical per-image dumps used a ground-truth ignore mask on the fit
half and do not store logits, so they cannot be corrected by replay alone.

Before generating Figure 6:

1. run the corrected evaluator on Runs A--C using the recorded checkpoints;
2. rerun the locked three-run, two-backend replay with one canonical risk grid per
   training run;
3. preserve fit-half selection and held-out-only reporting;
4. regenerate quality versus directly measured end-to-end route cost from canonical
   machine-readable artifacts;
5. report operating cells as correlated operating points, not independent samples.

Use `scripts/run_router_deployment_matched_pipeline.py` to produce the corrected
artifacts without overwriting historical evidence. The plotting script should be
added only after those artifacts exist. It must
parse those artifacts directly and must not copy the previous 120-cell numbers into
source code.
