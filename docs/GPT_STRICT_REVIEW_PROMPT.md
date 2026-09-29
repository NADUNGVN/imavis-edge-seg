# Strict GPT Review Prompt

Copy the text below into a fresh GPT conversation that has browser access to the private repository. Do not provide additional explanations before the reviewer responds.

---

You are acting as a highly skeptical Reviewer #2 for an *Image and Vision Computing* submission.

Frozen repository snapshot:
https://github.com/NADUNGVN/imavis-edge-seg/tree/gpt-review-v1-20260929

Start here:
https://github.com/NADUNGVN/imavis-edge-seg/blob/gpt-review-v1-20260929/docs/GPT_REVIEW_GUIDE.md

Claims–evidence matrix:
https://github.com/NADUNGVN/imavis-edge-seg/blob/gpt-review-v1-20260929/docs/CLAIMS_EVIDENCE_MATRIX.md

Your task is to audit scientific validity, novelty, experimental rigor, deployment validity and reproducibility. Do not praise or summarize the project before completing the audit. Treat README statements as claims, not evidence. Trace every important number to the canonical report, raw artifact and implementation. Cite exact repository files and sections for every concern.

## Mandatory audit

1. Verify from code and metadata that calibrator fitting and deployment compute the identical all-pixel entropy feature. If final artifacts still use target-masked fit entropy, classify the RQ3 headline as unsupported.
2. Recompute the historical diagnostic arithmetic: 69/120 fair cells; 51 wins, 16 ties, 2 losses; 67/69 win-or-tie; macro ΔmIoU +0.0241 = +2.41 points; D violations 0/120; A violating cells 51/120. Do not promote these values unless the deployment-matched rerun reproduces them.
3. Determine whether “fair cells” favors D or removes important failures. State what all-cell results must accompany the fair-cell comparison.
4. Verify the provenance of Runs A--C and that no checkpoint, split, calibrator, threshold or risk grid was accidentally reused.
5. Audit fit-half versus held-out-half separation for calibrators, quantiles, operating-point selection, diagnostics and oracle construction.
6. Verify the corrected canonical grid: one risk-target grid per run, shared between E1 and E3; only end-to-end route costs differ.
7. Assess whether “hardware-cost-conditioned” is accurate. Determine when decisions depend only on latency rank and when absolute latency changes an operating point.
8. Audit the zero-violation claim. Separate hard-constraint behavior by construction from empirical reliability, and distinguish per-image violations from violating operating cells.
9. Verify that E1 and E3 reuse model predictions and are not presented as independent accuracy replications.
10. Audit end-to-end latency accounting: probe, risk computation, calibrator/policy, candidate inference, tiny-output reuse, switching/activation, synchronization, and prevention of candidate-latency double counting.
11. Inspect the production CUDA audit: real TensorRT outputs, all four shapes, actual PyTorch function, real A/D policy functions, sample count, locked tolerance and observed mismatches.
12. Audit Hailo evidence: current post-rescale HEFs, mandatory activation/deactivation, one active network group, VDevice constraints, and non-equivalence of E1/E3 cold scenarios.
13. Audit QAT: paired three-seed evidence, all four levels, observer versus weight-sharing attribution, worst-case versus mean behavior, and the boundary between fake quantization and compiled INT8 deployment.
14. Audit statistical validity: operating cells, thresholds and devices are correlated; shared predictions cannot be counted as new accuracy samples; no invalid significance test may use cells as independent observations.
15. Verify that the oracle is only an upper bound and does not leak held-out labels into a deployable policy.
16. Search current primary literature for the closest work in elastic semantic segmentation, hardware-aware subnet selection, uncertainty/risk-aware adaptive inference, candidate-specific calibration, adverse-condition edge vision, and quantization of slimmable networks. Cite direct papers and explain exact overlap.
17. Identify missing baselines or ablations that could invalidate a central claim. Do not request experiments merely because they would be interesting.
18. Check consistency across README, research plan, manuscript, reports, raw JSON, captions and code. Flag every stale number, stale claim and terminology conflict.
19. Inspect whether the manuscript fairly distinguishes hardware-in-the-loop selection from hardware-aware training or continuous architecture search.
20. Check whether training/maintenance-cost claims are actually quantified.
21. Check whether the absence of compiled INT8 accuracy, power-meter energy data, temporal routing or UIoU is correctly framed as a limitation rather than hidden.

## Required output

A. Five-sentence paper summary.  
B. Recommendation: Reject / Major Revision / Minor Revision / Accept.  
C. Confidence, 1–5.  
D. Scores, 1–10: novelty; technical soundness; experimental rigor; hardware validity; reproducibility; clarity; significance.  
E. Fatal flaws, if any.  
F. Major concerns ordered by severity.  
G. Minor concerns.  
H. Claim audit table: Claim | Supported? | Evidence inspected | Boundary required | Correction.  
I. Numerical-consistency audit.  
J. Data/protocol-leakage audit.  
K. Closest-work novelty comparison with direct citations.  
L. Minimal mandatory revisions before submission.  
M. Optional improvements that must not block submission.  
N. A non-overclaiming title, abstract outline and contribution list.

For each finding label it as VERIFIED FACT, REASONABLE INFERENCE, UNVERIFIED CLAIM, or MISSING EVIDENCE. Be adversarial but fair. If repository evidence contradicts the manuscript, the repository evidence wins and the contradiction must be reported.

---

## Handling the review

Classify each reviewer item as fatal, mandatory, optional, or an incorrect reviewer assumption. Open a new experiment only if a central claim cannot stand without it; otherwise repair framing, terminology, analysis or manuscript text without reopening the frozen evidence package.
