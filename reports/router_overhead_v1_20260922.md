# Router end-to-end overhead measurement — v1, 2026-09-22 (E3 only, E1 pending)

Closing requirement 2 (`docs/COORDINATION_LOG.md` open thread #2), protocol locked
with Codex 2026-09-22. This report covers **E3 (AGX Xavier, TensorRT/CUDA) only** —
E1 (Raspberry Pi 5 + Hailo-8) is currently unreachable (Tailscale shows the device
offline, last seen 2 days; not a credentials/account issue — the account already
sees the device in its peer list) and still needs to be measured before this
closing requirement can be called fully satisfied per Codex's "validated on two
representative hardware backends" scope.

Measured directly on E3 via a self-contained harness
(`scripts/measure_router_overhead.py` locally, deployed and run via SSH — no
`imavis_edge_seg` package install needed on the device, only tensorrt/pycuda/numpy)
against real compiled TensorRT engines (untrained weights — legitimate for pure
latency measurement, same convention already used for E2/E5's original benchmark).

## Setup notes (real infrastructure work, not just measurement)

- E3 had no Python dev headers, no pip, no pycuda, no repo checkout. Set up from
  scratch over SSH: installed `python3.8-dev`/`python3.8-venv` (sudo, user-provided),
  bootstrapped pip via `get-pip.py` (apt's own `python3-pip` failed to resolve),
  built `pycuda==2022.2.2` from source against CUDA 11.4 (pycuda's *latest* PyPI
  release uses Python 3.10+ union-type syntax that breaks on this device's Python
  3.8), and worked around TensorRT 8.5's `__init__.py` still referencing the
  removed `np.bool` alias (monkeypatched before import). ONNX exports (untrained
  supernet, all 4 levels) built locally and `scp`'d over; TensorRT engines built via
  `trtexec --buildOnly` directly on E3 (note: this JetPack's `trtexec` doesn't have
  a `--skipInference` flag, unlike some other TensorRT builds — `--buildOnly` is the
  correct one here).
- **Two real bugs caught and fixed during smoke-testing, before trusting any
  number**: (1) the TensorRT output buffer is a flat 1D array; an early version of
  the risk-score function reshaped it as `(1, N)` and computed softmax over the
  *entire* ~1.4M-element array instead of the actual `(num_classes, H, W)` structure
  — wrong math and ~9x slower. (2) after fixing the reshape, a `.T` transpose to
  `(H*W, num_classes)` before the softmax reduction made every subsequent numpy op
  run on a non-contiguous view, another real slowdown; recomputing along the
  natural `(num_classes, H*W)` axis (no transpose) fixed this specific issue, but
  did **not** fully explain the remaining cost (see next finding).

## Finding 1: naive host-side (numpy/CPU) risk-score computation is a genuine,
severe bottleneck on this hardware

With the reshape/transpose bugs fixed, computing the risk score (softmax entropy
over ~1.4M elements) via numpy on E3's ARM CPU still costs **~40-80ms per call**
(isolated: `np.exp` ~25-33ms, `np.log` ~28-48ms, confirmed via direct microbenchmark
on synthetic arrays, unrelated to TensorRT). This is not a code bug — numpy's build
on this device has OpenBLAS + NEON/ASIMD SIMD support, but BLAS doesn't accelerate
elementwise transcendental functions (`exp`/`log`), and this numpy build's
vectorized-math support for those specific ops on ARM appears weak. **This is a
genuine, reproducible platform characteristic, not an implementation error.**

Consequence: with this "naive" host-side approach, the router's own decision
overhead (~45-80ms) is **5-50x larger** than the actual candidate-latency
differences it's trying to exploit (LUT-only: 0.94-9.41ms across all 4 levels on
E3) — the router's overhead would completely dominate and erase any benefit from
hardware-aware level selection, if this were the deployed implementation.

**However**, the project's actual `router.risk_probe.compute_risk_score` computes
entropy on a `torch.Tensor` that is still GPU-resident (the PyTorch model's raw
output), not on a numpy array copied back from a separately-compiled TensorRT
engine. The naive-numpy path measured here is representative of an *unoptimized*
TensorRT deployment (copy everything back to host, use numpy), not necessarily of
how this would actually be shipped.

## Finding 2: a GPU-resident entropy kernel closes almost all of this gap

Implemented a small custom CUDA kernel (via `pycuda.compiler.SourceModule`) that
computes per-pixel softmax entropy directly from the TensorRT engine's
device-resident output buffer — no full-tensor host copy, only the small
per-pixel entropy array (H×W floats, not `num_classes`×H×W) comes back to host for
the final mean. This is the GPU-resident design that matches the project's actual
`compute_risk_score`.

| route class | numpy backend (median, ms) | GPU-kernel backend (median, ms) | speedup |
|---|---|---|---|
| tiny→tiny | 43.38 | 2.07 | 21x |
| tiny→small | 56.57 | 5.23 | 10.8x |
| tiny→medium | 62.88 | 11.97 | 5.3x |
| tiny→large | 65.49 | 26.70 | 2.5x |

(n=500 per cell, all bootstrap 95% CI widths <0.5% of the median — very stable, no
throttling: 32.0°C → 38.5°C → 37.5°C across the whole run.)

**All numbers below use the GPU-kernel backend** — the naive-numpy numbers above are
reported as a real, cautionary finding (a naive TensorRT+numpy deployment would be
badly overhead-dominated), not as the number carried forward into the frontier
analysis.

## Real end-to-end latency, GPU-kernel backend (the numbers used for the frontier
replay below)

| route class | warm/resident (median, ms) | cold/reload (median, ms) |
|---|---|---|
| tiny→tiny | 2.07 | n/a (no reload) |
| tiny→small | 5.23 | 22.74 |
| tiny→medium | 11.97 | 42.63 |
| tiny→large | 26.70 | 76.95 |

Cold/reload costs 2-3x the warm/resident number, as expected (genuine
context-destroy-and-rebuild-from-file per transition, 50 iterations each). All 4
engines' memory footprint is trivial on this device (14GB total RAM, engines
343KB-2.4MB each) — warm/resident (all 4 engines concurrently loaded) is easily
feasible here, no cache-policy compromise needed.

Decision-path microbenchmark (2000 iterations each, CPU/numpy only — no GPU
involved, self-contained reimplementations of the real calibrator/policy logic):
cell A (single calibrator + rank policy) median **0.053ms**; cell D (per-candidate
calibrators + budget-constrained policy) median **0.201ms**. Both negligible next
to the probe/candidate inference and dispatch costs above — real, but not the
bottleneck.

## Frontier replay: A vs. D under real e2e latency (Cityscapes, E3, seed0)

Using the already-locked seed0 held-out result
(`reports/router_E3_20260921.json`), for each strategy's previously-chosen
operating point (fit-half-selected `risk_target`, per the original LUT-only
budget), its held-out **routing distribution** (count of images routed to each
level) is reweighted using the real GPU-kernel warm e2e latency per route class,
instead of the pure LUT-only candidate latency:

| strategy | orig. (LUT-only) budget | routing distribution | **real e2e latency** | achieved mIoU |
|---|---|---|---|---|
| A (`calibrated_risk`) | 0.941 | tiny:241, small:9 | **2.18ms** | 0.3055 |
| A | 1.796 | tiny:192, small:58 | **2.80ms** | 0.3268 |
| A | 4.607 / 9.414 (same point) | small:143, medium:98, large:9 | **8.65ms** | 0.4251 |
| D (`risk_latency_constrained`) | 0.941 | tiny:250 | **2.07ms** | 0.3003 |
| D | 1.796 | small:250 | **5.23ms** | 0.3701 |
| D | 4.607 | medium:224, small:26 | **11.27ms** | 0.4369 |
| D | 9.414 | large:188, medium:36, small:26 | **22.35ms** | 0.5240 |

**Reading**: even after substituting real, measured end-to-end latency for the
pure LUT-only candidate numbers, **D still produces a wider, more useful
quality-latency frontier than A**. A's operating points collapse to only 3
distinct points and saturate at 0.4251 mIoU (its own fit-half-based selection
procedure runs out of usable `risk_target`s once the nominal budget grows past
what its rank-step policy can exploit); D reaches a 4th point (22.35ms, 0.5240
mIoU) that is **not achievable by A at any latency A actually uses** — the
project's largest achievable quality remains exclusive to D even under real
overhead. At the very cheapest end, A's first point (2.18ms, 0.3055) is very
slightly better than D's (2.07ms, 0.3003) — consistent with the earlier
(LUT-only) finding that D and A are close/tied at the tightest operating point.

**Important limitation on this replay, stated plainly**: this substitutes real
e2e latency into the *existing* routing distributions (computed under the
original LUT-only budget grid and selection procedure) — it is **not** the full
per-image replay Codex's protocol specifies (map every held-out image to its
own route class, recompute the entire at-budget selection *using the new
e2e-based budget grid*). The persisted `reports/router_E3_20260921.json` doesn't
retain per-image routing decisions or fit-half routing distributions, only
aggregated held-out counts per risk-target operating point — a full redo would
need a rerun of `evaluate_router.py` with per-image logging added, and a
fresh selection of representative operating points against a **new,
e2e-based** device-budget grid (e.g. `[2.07, 5.23, 11.97, 26.70]` instead of the
old `[0.941, 1.796, 4.607, 9.414]`), which has not been done yet.

## Update, 2026-09-22: full per-image, e2e-aware replay — CONFIRMED (E3
provisional pass)

Per Codex's review of the approximation above ("not sufficient to close the
requirement" — needs the policy's own budget-feasibility decision re-run
against the real latency table, not just relabeled after the fact), built
`scripts/evaluate_router.py --dump-per-image` (persists, per held-out AND
fit-half image: raw probe score, each level's own ground-truth confusion
matrix, and the fitted per-level calibrators — enough to replay any re-routing
exactly, without re-inference) and `scripts/replay_router_with_overhead.py`
(recomputes the actual policy *decision*, not just its score, in two strictly
separate modes).

**Post-hoc replay** (decision recomputed exactly as the original LUT-only run,
scored with real e2e latency): confirms the LUT-only latency model was badly
wrong in absolute terms — nearly every operating point now "violates" its
original LUT-based budget (real e2e latency, 2.07-26.70ms, exceeds every
original LUT budget, 0.94-9.41ms). This number alone says nothing about D vs.
A — it only shows the old latency assumption was wrong, exactly as expected.

**E2E-aware replay** (decision recomputed using real e2e latency as both the
ranking/budget-check input and the new device-budget grid `[2.07, 5.23, 11.97,
26.70]` ms; representative operating point re-selected via `select_budget_
matched_operating_point` on fit-half stats only, never touching held-out
before that lock — the real deployment result): **confirms the earlier
approximation**.

| | value |
|---|---|
| Total cells (5 splits × 4 e2e budgets) | 20 |
| Fair cells (A itself doesn't violate) | 12/20 |
| D vs. A on fair cells | **8 wins, 3 ties, 1 loss** (67% win rate) |
| Mean (D − A), fair cells | **+0.0207** |
| Mean (D − oracle), fair cells | −0.0057 |
| **D budget violations, all 20 cells** | **0/20** |
| A budget violations, all 20 cells | 8/20 (mean violation rate 8.95%) |

The single loss (`acdc/rain`, budget=5.23ms: A=0.3465 vs. D=0.3238) reproduces
the same isolated pattern seen in the approximation. D again reaches a quality
level (0.5240 mIoU, `cityscapes` at the largest budget) that A cannot reach at
*any* budget A uses (A caps at 0.4251 on `cityscapes` regardless of budget).
**This is now the rigorous, per-image confirmation Codex required, not an
approximation** — E3 is a provisional pass (still pending E1 per the
two-backend scope).

## Not yet done

- **E1 (Hailo-8) measurement** — device currently offline (Tailscale confirms
  this is a connectivity/power issue, not a credentials/account problem — the
  account already sees the device in its peer list, just marked offline).
  Codex's "two representative backends" scope means this closing requirement
  is not yet fully satisfied with E3 alone; E3 results cannot be extrapolated
  to Hailo (the CUDA entropy kernel doesn't transfer; Hailo's own host-side
  cost structure is unknown until measured).
- Cold/reload frontier replay (only warm/resident was used for the analysis
  above, per Codex's rule that cold is a stress-test/robustness number, not
  automatically gating the closing decision).
- Separating the naive-numpy path's ~40-80ms into (device-to-host transfer/
  sync) vs. (numpy softmax/entropy) vs. (calibrator/policy) components, per
  Codex's request — not yet done; currently reported only as "naive host-side
  risk evaluation path" cost, not attributed solely to numpy's `exp`/`log`.
- Validating the GPU-kernel risk score against the reference (PyTorch)
  implementation within a locked tolerance, and confirming candidate decisions
  (A/D) are unchanged between the numpy and GPU-kernel backends except
  arithmetic ties — not yet done as an explicit check beyond the smoke-test
  sanity check already in this report.
