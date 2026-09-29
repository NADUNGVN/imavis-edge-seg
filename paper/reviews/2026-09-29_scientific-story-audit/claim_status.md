# Claim status after scientific-story revision

| Claim area | Status | Evidence boundary |
|---|---|---|
| Compiler-safe elastic candidates | Supported deployment property | Four static graphs compile and execute; no trained unconstrained-operator control |
| RQ1 measured cost | Ready | Four candidates, four hardware targets, proportional through-origin FLOPs proxy only |
| RQ2 shared-training quality | Ready as descriptive near-parity | Three augmented runs per model; no claim of superiority or lifecycle savings |
| RQ3 candidate-specific routing | Conditional diagnostic | Canonical 120-cell replay retained, but fit and deployment entropy use different pixel sets |
| Cross-backend routing | Deployment-cost comparison only | E1 and E3 share prediction caches and are not independent accuracy replications |
| Budget violations | Descriptive constraint behavior | D's zero count is partly structural and not empirical reliability |
| QAT | Secondary fake-quant evidence | No compiled INT8 accuracy or speedup claim |
| Energy | Unsupported and removed | No external power measurement |

## Canonical conditional routing diagnostic

- 120 operating cells, of which 69 are fair by A's feasibility.
- D versus A on fair cells: 51 wins, 16 ties, and 2 losses.
- Macro D-minus-A: +0.0241 mIoU.
- Full-grid budget-violating cells: D 0/120 and A 51/120.

These correlated operating cells are not statistical replicates. The values remain traceable evidence but are not used as the abstract or conclusion headline in manuscript v2.

## Submission gate

Compute the calibration feature identically during fitting and deployment, rerun policy fitting and held-out replay without changing the locked split protocol, and then audit the revised RQ3 result before promoting it to a headline claim.
