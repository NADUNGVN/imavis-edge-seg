# 07 — Novelty and related work

## Verification status of key references
Bibliographic checks performed in this session:
- **Verified online (arXiv abstract page):** Haberer et al., *Slimmable ConvNeXt*, arXiv:2605.22677 (2026).
- **RADLER:** the only RADLER found (arXiv 2504.12167) is radar object detection, not slimmable segmentation → not cited. No other RADLER paper was found; the author should supply the reference if a different one is meant.
- **Not verified online in this session** (bib entries inspected for completeness only): MESS (Kouris et al., ECCV 2022), Dynamic Routing (Li et al., CVPR 2020), Once-for-All (Cai et al., ICLR 2020), SlimSeg (Xue et al., ACM MM 2022, DOI present), AutoSegEdge (Dou et al., IVC 2023, DOI present), OffSeg (Zhang et al., ICCV 2025). These are well-known venues, but Table 2 characterizations ("exit latency", "compute penalty", etc.) should be confirmed against the original papers before submission.
- Bib formatting is inconsistent (initials vs full names; some entries lack DOI). MINOR.

## Positioning assessment
| Prior work | What it does | Overlap with PACE-Seg | Distinct here |
|---|---|---|---|
| MESS | multi-exit segmentation, per-image exit choice under latency | per-image capacity choice | static separately compiled engines; same-harness static baseline; Hailo-8 |
| Dynamic Routing | learned in-graph path selection, compute penalty | input adaptivity | measured accelerator cost; static engines |
| Once-for-All | elastic supernet, design-time specialization with latency predictor | elastic training | run-time selection; measured routes |
| SlimSeg | slimmable segmentation widths | elastic capacities | per-image selection; deployment measurement |
| Learned latency predictors (nn-Meter, HW-NAS-Bench) | predict latency | cost modelling | direct measurement of 4 routes; RQ1 only tests a proportional proxy |

**Novelty statement that the evidence supports:** a measured, same-harness comparison of input-adaptive capacity routing against static deployment on TensorRT and Hailo-8, with an ablation of router ingredients and a break-even overhead analysis. The paper does **not** support novelty claims for the router itself (T-hard is as good). Current V20 text is consistent with this.

**SUGGESTED IMPROVEMENT:** cite one or two works that report adaptive-vs-static wall-clock comparisons or switching overhead on accelerators, if they exist (search not completed in this session).
