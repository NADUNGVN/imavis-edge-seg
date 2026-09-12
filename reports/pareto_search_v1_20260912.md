# Pareto subnet selection v1 — 2026-09-12

Implemented `src/imavis_edge_seg/search/pareto.py` (Contribution 2): joins real
measured latency (`benchmark/lookup_table.py`'s records) with real measured mIoU
(`evaluate_supernet.py`'s output), computes the Pareto frontier per deployment target
(device + backend), and picks the best level under a latency budget. This is subnet
*selection* among the 4 already-trained elasticity levels, not a continuous
architecture search -- that already happened at training time via the supernet's
width/depth axes.

9/9 new tests pass (synthetic dominance/frontier/budget-selection logic, plus a
real-data join test using the actual 2026-09-10/11 large-level numbers). 74/74 total.

## Ran against our real data end-to-end (`scripts/build_pareto_frontier.py`)

Cityscapes mIoU x p95 end-to-end latency, both backends, seed-0 supernet:

**E1 (Hailo)**: tiny 3.72ms/0.2986 -> small 6.63ms/0.3564 -> medium 20.47ms/0.4120 ->
large 41.18ms/0.4712 -- all 4 levels are Pareto-optimal (monotonic in both axes, no
crossovers, matches every prior mIoU table). Best under a 10ms budget: **small**.

**E3 (TensorRT GPU)**: tiny 0.94ms -> small 1.80ms -> medium 4.61ms -> large 9.41ms,
same mIoU values. All 4 levels Pareto-optimal here too. Best under the *same* 10ms
budget: **large**.

## Why this matters (concrete illustration of RQ1)

The identical 10ms latency budget picks a *different* level depending on which device
it runs on -- `small` on E1, `large` on E3 -- because E3's TensorRT GPU is roughly
4-4.5x faster per level than E1's Hailo at this model size. A single FLOPs-based or
one-size-fits-all subnet choice cannot capture this; it requires the actual measured,
per-device cost this module consumes. This is the first concrete (if minimal)
evidence for RQ1's premise, not yet a full test of RQ1's hypothesis (needs a FLOPs-
aware baseline selection to compare against, and multiple datasets/conditions).

## Not yet done

- No FLOPs-aware baseline selection to compare against (RQ1 needs "measured beats
  FLOPs-aware by >=15-20%" -- requires computing/estimating FLOPs per level, not just
  params, and running the same budget-selection logic against that instead).
- Only 2 backends (E1, E3) have latency data; E2/E5 (TensorRT) and DLA (secondary
  ablation) aren't in the lookup table yet.
- No energy axis (deferred, no power meter yet) -- `mu_energy` term from
  `docs/RESEARCH_PLAN.md` §5.4's objective is unused; this is latency-only selection.
- Not yet wired into the router (Contribution 3) or into an actual deployment
  decision-maker -- this is the underlying selection logic, not the calibrated-risk
  policy that will eventually call it.
