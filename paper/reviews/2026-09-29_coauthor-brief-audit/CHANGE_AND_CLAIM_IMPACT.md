# Change and claim impact

## Scientific correction

The router evaluator formerly supplied target-masked entropy to the calibrator on
the fit half and all-pixel entropy on held-out/runtime inputs. The corrected API
accepts logits only and computes all-pixel entropy for both phases. Ground-truth
masking remains confined to the supervised observed-error target.

Regression coverage verifies that the deployment feature is identical at fit and
inference and differs from the legacy masked feature when an ignored pixel has
distinct entropy.

The progressive evaluator also selected D's representative risk target from
held-out D metrics, despite the final overhead replay using the correct fit-half
selection. The evaluator now records fit and held-out metrics separately and selects
all representative operating points from fit-half metrics only. Historical final
replay arithmetic is unchanged; historical progressive-ablation output is not final
evidence.

A pooled-across-conditions candidate calibrator was added as a low-cost baseline.
It reuses the same cached predictions and therefore adds no model inference. The
deployment-matched rerun will compare configured condition-specific D against this
condition-agnostic calibration baseline.

## Claim impact

- RQ1 is unchanged.
- RQ2 is unchanged numerically; wording is narrowed to observed practical
  near-parity within the prespecified band.
- RQ3 mechanism remains part of the method.
- Existing RQ3 numbers are not final evidence because they depend on the old feature.
- Figure 6 remains intentionally absent until the corrected Run A/B/C pipeline is run.
- QAT remains secondary fake-quant evidence.

## Presentation cleanup

The main manuscript now uses Run A, Run B, and Run C. Internal experiment IDs are
restricted to Data and Code Availability and the evidence manifest. QAT rows use the
same labels.

## External blocker

The local workspace contains neither the three 100,000-step checkpoints nor the
licensed Cityscapes/ACDC image data. The code correction, replay chain, metadata,
tests, and manuscript boundaries can be completed locally; the quantitative RQ3
rerun requires a training server with those inputs.
