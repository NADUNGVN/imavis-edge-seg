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

## Update, 2026-09-28: E1 (Hailo-8) measured for real — CONFIRMED, closing requirement 2 now satisfied on both backends

E1 (Pi5 + Hailo-8, alias `pi5`) came back online 2026-09-28 (a real
connectivity/power issue, not an account problem, as already suspected).
Direct-SSH setup and measurement, same pattern as E3: `hailo_platform`
(pyhailort) is already installed system-wide on E1 (HailoRT 4.23.0), so no new
package install was needed there. The HEFs used for the compiler smoke test
back in week 1-2 were the old ~5K-parameter placeholder architecture (pre the
2026-09-10 model rescale) and are no longer resident on the device (`/tmp` is
wiped on reboot) — **recompiled all 4 levels fresh from the current,
post-rescale ONNX exports** (`outputs/onnx_e2e5/pace_seg_{level}.onnx`, the
same exports E2/E5's TensorRT engines use) via the cached Hailo Dataflow
Compiler 3.34.0 toolchain in WSL2 (`hailo parser` → `hailo optimize
--use-random-calib-set` → `hailo compiler`, all 4/4 parse→optimize→compile
PASS), so E1's engines are the same architecture size as E3's — a fair
comparison. New harness: `scripts/measure_router_overhead_hailo.py`, built
from scratch against the real `hailo_platform` Python API (VDevice/HEF/
ConfigureParams/InferVStreams — explored live via disposable smoke scripts
before writing the real harness, same discipline as E3).

**A second real, load-bearing platform finding — different from E3's:**
unlike E3 (where naive host-side numpy entropy computation dominated
overhead), **this Hailo-8 M.2 module only allows one network group
hardware-activated at a time** — every single inference call, even in the
warm/resident scenario, pays a real `activate()`/`infer`/`deactivate()` cycle
(confirmed live: calling `infer()` on a non-activated network group raises
`HailoRTNetworkGroupNotActivatedException`). This activate/deactivate cost,
not entropy computation, is what dominates E1's real overhead. There is no
GPU-resident-kernel equivalent fix here — Hailo's dataflow architecture has no
general-purpose compute-shader model like CUDA, so **only a numpy/host entropy
backend exists on this device, by hardware necessity, not because the
GPU-kernel optimization step was skipped**. Report `report["warm"]["numpy"]`
only; do not compare against E3's `report["warm"]["gpu"]` numbers directly —
`scripts/replay_router_with_overhead.py --entropy-backend` was extended to
select which backend's e2e latency to replay against precisely for this
reason.

Two further real infra findings while building the cold/reload scenario
(caught the same way as E3's bugs — by the numbers looking wrong, not by
inspection): (1) `ConfiguredNetwork` (a HEF configured onto a `VDevice`) has
**no public release/deconfigure method at all** — `del` plus `gc.collect()`
does not free its slot either; a `VDevice` accumulates configured network
groups for its entire lifetime, capped at 32 "core-ops" on this chip, so the
first cold-scenario design (candidate reloaded repeatedly onto the same
long-lived `VDevice` that also held the resident probe) hit
`HAILO_INVALID_OPERATION` partway through after ~28 accumulated reloads. (2)
this Hailo-8 module allows **exactly one `VDevice` at a time** — a second,
concurrent `VDevice` (tried as a workaround, to keep the probe on one handle
and reload candidates on another) raises `HAILO_OUT_OF_PHYSICAL_DEVICES`
immediately. Both are genuine hardware/driver constraints of this
single-context, no-scheduler configuration, not script bugs. Fix: E1's cold
scenario reconfigures the **entire** `VDevice` fresh every iteration — probe
included, even for `tiny->tiny` — which is a stricter (more pessimistic) cold
number than E3's candidate-only reload; the report and script docstrings
disclose this design difference explicitly rather than presenting it as the
same measurement.

**Full protocol run, real numbers (`reports/router_overhead_E1_20260928.json`):**

| Route class | LUT-only (hailortcli streaming) | Real e2e, warm (numpy) | Real e2e, cold (full reconfigure) |
|---|---:|---:|---:|
| tiny→tiny | 3.72 ms | 34.95 ms | 120.0 ms |
| tiny→small | 6.63 ms | 46.13 ms | 156.0 ms |
| tiny→medium | 20.47 ms | 63.36 ms | 176.0 ms |
| tiny→large | 41.18 ms | 92.68 ms | 214.6 ms |

All warm route classes converged at the base 500 measured iterations (CI
width ≤1.2% of the median, well under the 2% adaptive-extension threshold —
no route class needed the adaptive extension to 2000). No throttling: SoC
temperature 49.6→55.1→54.6°C, Hailo-8 chip telemetry (a real on-chip read via
`hailortcli`-equivalent `get_chip_temperature()`, same "not an external
calibrated meter" caveat as the existing E1 benchmark-protocol note)
46.1→48.5→47.7°C across the whole run. Decision-path microbenchmark
(hardware-agnostic, CPU-only, expected to match E3): A=0.012 ms, D=0.045 ms —
matches E3's order of magnitude, as expected.

**Full per-image, e2e-aware replay
(`reports/router_overhead_replay_E1_20260928.json`), same locked methodology
as E3's:**

| | value |
|---|---|
| Total cells (5 splits × 4 e2e budgets) | 20 |
| Fair cells (A itself doesn't violate) | 12/20 |
| D vs. A on fair cells | **8 wins, 3 ties, 1 loss** (67% win rate) |
| Mean (D − A), fair cells | **+0.0207** |
| Mean (D − oracle), all cells | −0.0036 |
| **D budget violations, all 20 cells** | **0/20** |
| A budget violations, all 20 cells | 8/20 (mean violation rate among violators 22.36%) |

**This is essentially an exact qualitative and near-exact quantitative
replication of E3's result** (E3: 8/3/1, mean(D−A)=+0.0207, D 0/20
violations, A 8/20 violations) despite E1's overhead being dominated by a
completely different mechanism (hardware activate/deactivate cycling vs.
host-side entropy computation) and E1's absolute latencies being 2-4× higher
than E3's at every level. The single loss and the specific fair/unfair cell
split are the same pattern as E3's, driven by the same underlying per-image
data (one shared dump) — the new information here is that the *real, measured
overhead* on a structurally different accelerator still doesn't erase D's
advantage, which is exactly what "validated on two representative hardware
backends" needs to mean.

**Closing requirement 2 (real end-to-end router overhead, E1 + E3) is now
CONFIRMED on both backends.** Admissible claim, extended: *"Candidate-specific,
device-conditioned routing (D) continues to win under real, directly measured
end-to-end overhead — including router-specific costs invisible to a pure
inference-latency lookup table — on two structurally different accelerator
backends (TensorRT/CUDA GPU and Hailo-8 dataflow NPU), despite those two
backends' overhead being dominated by different mechanisms."*

## Update, 2026-09-28: GPU-kernel correctness audit — PASS, mandatory pre-lock requirement closed

Codex's explicit, mandatory (not optional) pre-manuscript-lock requirement:
*"E3 CUDA entropy/risk kernel phai duoc doi chieu voi PyTorch reference truoc
manuscript lock"*, with a locked tolerance covering (1) risk-score
absolute/relative error, (2) route-decision agreement rate, (3) tie-handling.
This is a **correctness audit, not a new experiment** — it isolates the CUDA
kernel's own arithmetic from any model-execution or backend difference by
feeding both implementations the *exact same* synthetic logits array, never
re-running inference. `scripts/audit_gpu_risk_kernel.py` (self-contained,
same discipline as the other harnesses): the "PyTorch reference" is the
already-duplicated numpy transcription of `router.risk_probe.
compute_risk_score`'s exact no-ground-truth formula (`F.softmax(dim=1)` →
`-(p·log(clamp_min(p,1e-12))).sum(dim=1)` → `.mean()`) — confirmed
line-for-line identical to the numpy version already shipping in
`measure_router_overhead.py`, so no new torch install was needed on E3 (torch
is never present on any edge device in this project). Output shapes read
directly from E3's real, on-disk `.engine` files (not hardcoded).

**Proposed tolerance, locked before running** (FP32 arithmetic over ~19
accumulated multiply-adds per pixel plausibly accumulates ~1e-5 to 1e-4
absolute error; entropy's own range is bounded by ln(19)≈2.94 nats, so these
are generous but not vacuous relative to the signal's scale):

| criterion | tolerance | observed (max over all 4 levels/200 trials) | pass? |
|---|---|---|---|
| per-pixel max abs error | ≤1e-3 nats | 9.54e-07 | **PASS** (~1000x margin) |
| scalar risk-score abs error | ≤1e-4 nats | 2.38e-07 | **PASS** (~400x margin) |
| decision_a agreement (2000 trials) | ≥99.9% | **100.000%** (0/2000 mismatches) | **PASS** |
| decision_d agreement (2000 trials) | ≥99.9% | **100.000%** (0/2000 mismatches) | **PASS** |

**Overall: PASS, with wide margin, at every real output shape (tiny/small/
medium/large) and both policies (A/D). Zero decision mismatches across 2000
trials means the tie-handling question is moot here** — the kernel and the
reference never disagreed on a route decision in this sample, so there was no
tie case to characterize separately. This is expected: the two
implementations run the *identical* formula (softmax + entropy), so any
disagreement can only come from floating-point non-associativity (different
summation order between the CUDA kernel's serial per-pixel loop and numpy's
vectorized reduction), which FP32 keeps well under 1e-6 at this array size.
Full raw output: `reports/audit_gpu_risk_kernel_E3.json`.

**This closes the last mandatory pre-manuscript-lock requirement from
Codex's 2026-09-28 review of closing requirement 2.** The GPU-kernel numbers
used throughout the E3 overhead analysis (and by extension the closing-
requirement-2 confirmation) can now be reported as numerically validated
against the reference implementation, not merely "smoke-tested."

## Update, 2026-09-29: cross-seed E2E replay — CONFIRMED across all 3 seeds, both backends (canonical-grid corrected)

Codex's instruction (2026-09-28): the 20-cell E2E-aware replay above used only
`pace_seg_v1_seed0`, while the router *method* was already confirmed on 3
seeds (closing requirement 1). Rather than re-measure hardware overhead
(device-level timing depends only on the destination level, not on which
checkpoint produced the logits — already established methodology), replay
the same E1/E3 overhead JSONs against fresh per-image dumps from
`pace_seg_v1_seed2` and `pace_seg_v1_aug_seed3` (`scripts/evaluate_router.py
--dump-per-image`, run on SERVER-02, same protocol as seed0's dump). No new
hardware measurement.

**A real bug, caught by Codex's review, not by this project's own checking**:
the first pass of this replay passed `--original-results-json
reports/router_{E1,E3}_20260921.json` (seed0-era files) for *every* seed,
including seed2/seed3 — so those two seeds' replays used **seed0's
`risk_target_grid`**, not their own. The fix, per Codex's explicit protocol:
lock ONE fit/held split and ONE `risk_target_grid` per seed, and use that
same canonical grid for **both** E1 and E3's replay (only the E2E latency
table changes per backend) — never let each backend independently re-fit a
grid that is methodologically meant to be device-independent. Concretely:
seed2/seed3 replays now use `reports/router_E3_seed{2,3}_dump_meta.json`
(the grid computed in the SAME `evaluate_router.py --dump-per-image`
invocation that produced that seed's per-image dump — self-consistent by
construction) for **both** devices. Seed0 turned out to already be canonical
by coincidence: `router_E1_20260921.json` and `router_E3_20260921.json`'s
`risk_target_grid` values are byte-identical (verified), so seed0's numbers
are unchanged by this fix.

**Canonical per-seed table:**

| device | seed | fair cells | W/T/L | win+tie | mean(D−A) | D violations | A violations |
|---|---|---|---|---|---|---|---|
| E3 | seed0 | 12/20 | 8/3/1 | 91.7% | +0.0207 | 0/20 | 8/20 |
| E3 | seed2 | 10/20 | 7/3/0 | 100.0% | +0.0241 | 0/20 | 10/20 |
| E3 | seed3 | 13/20 | 11/2/0 | 100.0% | +0.0273 | 0/20 | 7/20 |
| E1 | seed0 | 12/20 | 8/3/1 | 91.7% | +0.0207 | 0/20 | 8/20 |
| E1 | seed2 | 10/20 | 7/3/0 | 100.0% | +0.0241 | 0/20 | 10/20 |
| E1 | seed3 | 12/20 | 10/2/0 | 100.0% | +0.0276 | 0/20 | 8/20 |

**Aggregated (Codex's requested summary format):**

| Backend | Fair cells | W/T/L | Win-or-tie | Macro ΔmIoU | D violating cells | A violating cells |
|---|---|---|---|---|---|---|
| E3 | 35/60 | 26/8/1 | 97.1% | +0.0241 | 0/60 | 25/60 |
| E1 | 34/60 | 25/8/1 | 97.1% | +0.0241 | 0/60 | 26/60 |
| **Overall** | **69/120** | **51/16/2** | **97.1%** | **+0.0241** | **0/120** | **51/120** |

(+0.0241 mIoU = **+2.41 mIoU points** on the conventional 0–100 scale — stated
both ways to avoid ambiguity.)

**Important scope note (Codex's explicit correction)**: 69/120 is the count
of *operating cells* eligible for a fair A-vs-D comparison, and 0/120 is the
count of D's operating cells with a budget violation — **these are not 120
independent statistical samples**. E1 and E3 replay the *same* per-image
model predictions (from the *same* 3 checkpoints) against two different real
latency tables; this is **cross-backend deployment validation**, not two
independent accuracy replications. seed2's identical stats between E3/E1 are
the clearest illustration: both A's rank-based escalation and D's budget
check reduce to comparisons between a chosen level's *rank* among the 4
device-specific latencies, not their absolute values, whenever the budget
grid is itself built from those same 4 per-device latencies — so identical
per-image decisions across backends are the *expected* outcome when latency
rankings agree, not evidence that D independently "discovers" different
per-device behavior. Concretely: **backend-specific costs determine the
deployed latency and feasible budget scale, while the breakpoint-based
evaluation can produce identical choices when candidate latency rankings
agree.** Do not read this as "D always chooses a different architecture per
device" — call the method **hardware-cost-conditioned** (or
**device-cost-conditioned**) routing, not "device-conditioned decisions."
Seed3 is the counterexample that shows this isn't purely mechanical: E3 (13
fair cells) and E1 (12 fair cells) diverge slightly there, traced to
`select_budget_matched_operating_point`'s fit-half *mean latency* (a
continuous quantity, in each device's own absolute ms) landing on different
sides of a budget threshold for one operating point — a legitimate,
expected, backend-value-dependent effect, not a bug.

**Updated admissible claim (Codex's template, adopted with the canonical
numbers substituted in — Codex's own provisional table was explicitly marked
"cập nhật lại nếu số thay đổi" pending this canonical-grid fix, and the
actual corrected counts differ slightly from Codex's provisional draft)**:

*"Using directly measured end-to-end route costs, candidate-specific,
hardware-cost-conditioned routing outperformed or matched the rank-based
policy in 67 of 69 fair operating cells (51 wins, 16 ties, 2 losses) across
three independently trained supernets and two structurally different
accelerator backends. It achieved a macro-averaged gain of 0.0241 mIoU (2.41
points) and produced no budget-violating operating point across all 120
evaluated cells."*

Boundary, stated immediately after the claim, every time it is used:

*"The two backend evaluations share model predictions and therefore
demonstrate cross-backend deployment robustness rather than independent
accuracy replication."*

System-level insight (also part of the claim's context):

*"The dominant routing overhead was backend-specific: host-side uncertainty
evaluation required accelerator-resident optimization on TensorRT/CUDA,
whereas mandatory network-group activation dominated Hailo-8."*

**Three annotations that must accompany this claim in the manuscript
(Codex's final-lock instruction, 2026-09-29 — never state the claim without
these)**:

1. **Operating cells are not statistically independent samples.** Never use
   n=120 (or n=69) as the sample size for a significance test or confidence
   interval — E1/E3 share the same per-image model predictions, and the
   5 splits/4 budgets within one device are correlated (same images,
   overlapping budget-feasible sets), not 120 (or 69) independent draws.
2. **Different denominators for different numbers**: W/T/L and the win-or-tie
   rate are computed over the **69 fair cells only** (cells where A itself
   respects its own stated budget); the budget-violation counts (D: 0/120,
   A: 51/120) are computed over **all 120 evaluated cells**, fair or not.
   Never blend these two denominators in prose.
3. **Zero violations is a constraint-satisfaction result, not the headline
   finding by itself** — it follows partly from D's own hard-budget-by-design
   policy (D only ever selects a candidate within its stated budget, by
   construction). The scientific claim is that **D keeps its quality
   advantage over A while satisfying that constraint**, not that "zero
   violations" alone is remarkable.

Raw data: `reports/router_overhead_replay_{E1,E3}_20260928.json` (seed0,
canonical already) and `reports/router_overhead_replay_{E1,E3}_seed{2,3}_
20260929.json` (seed2/seed3, corrected 2026-09-29 to use the canonical
per-seed grid). Per-image dumps: `reports/router_per_image_dump_seed{2,3}
.json`, canonical grid source `reports/router_E3_seed{2,3}_dump_meta.json`.

## Update, 2026-09-29: CUDA-kernel-vs-production-PyTorch audit on real engine outputs — PASS

Codex's second correction to the 2026-09-28 audit: comparing the CUDA kernel
against a **numpy transcription** of `compute_risk_score`'s formula, on
**synthetic** logits, is good arithmetic evidence but "chưa hoàn toàn tương
đương" a real PyTorch-reference audit. Two closes, both real:

1. **Real production import**: `scripts/audit_gpu_risk_kernel_production.py`
   calls `imavis_edge_seg.router.risk_probe.compute_risk_score` directly (the
   actual function, no transcription) as the reference.
2. **Real engine outputs, not synthetic logits**: `scripts/
   capture_real_engine_outputs.py` (run on E3) performs real TensorRT
   inference for **40 real outputs (10 samples × 4 levels)** and, per
   sample, (a) runs the exact deployed GPU-kernel path (`infer_no_copy` +
   `risk_score_gpu`, reading `device_out` directly — no intermediate host
   copy, so this also validates the real integration, not just kernel
   arithmetic in isolation) and (b) a *second, independent* `infer_sync`
   call to capture the full host-side array for the reference computation
   (the two capture paths can't leak into each other). 434MB of real
   captured output arrays transferred to the dev machine (not committed to
   git — reproducible via the capture script with a fixed seed) and fed
   through the real `compute_risk_score`.

Per Codex's explicit framing (2026-09-29): the earlier **2,000 synthetic
trials** (`audit_gpu_risk_kernel.py`) are an **arithmetic stress test** —
broad coverage of the entropy formula's numerical behavior, not tied to any
one real model. These **40 real TensorRT outputs** are a separate, narrower
**integration audit** — they exercise the real on-device buffer layout,
dtype, synchronization, and indexing path end-to-end, which only a real
engine run can expose. The two are complementary, not redundant, and neither
substitutes for the other.

| criterion | N | tolerance (locked before running) | observed (max) | pass? |
|---|---|---|---|---|
| scalar risk-score abs error | 40 (10 samples × 4 levels) | ≤1e-4 nats | 7.15e-07 (large level) | **PASS** (~140x margin) |
| decision_a agreement | 10 (tiny level, the probe) | ≥99.9%* | **100.000% (0/10 mismatches)** | **PASS** |
| decision_d agreement | 10 (tiny level, the probe) | ≥99.9%* | **100.000% (0/10 mismatches)** | **PASS** |

\* 99.9% is the **pre-registered acceptance threshold**, not a statistical
estimate of the true agreement rate derived from N=10 or N=40 — these sample
sizes are far too small to resolve a rate that precisely; 0/10 and 0/40
mismatches is the entire evidentiary claim, reported as exact counts, not as
an inferred percentage with a confidence interval.

Zero mismatches on all 10 real decision trials. Decision-agreement here also
uses the real production
`router.calibrator.fit_risk_calibrator`/`RiskCalibrator.predict` and
`router.policy._select_by_risk`/`_select_by_risk_and_latency_budget`
functions (not reimplementations) — closes the last piece of Codex's
"CUDA-versus-NumPy formula audit, not a PyTorch-reference audit" distinction.
Raw: `reports/audit_gpu_risk_kernel_production_E3.json`.

**Both Codex-mandated correctness items from the 2026-09-29 review are now
closed. The evidence package for router closing requirement 2 is complete and
canonical.**

## FINAL LOCK, 2026-09-29 — Codex

Closing requirement 2 is **PASS and permanently locked**: canonical
per-seed risk grid (shared identically between E1/E3, only the hardware
cost table differs), 3 supernet seeds, 2 structurally different accelerator
backends, directly measured end-to-end route costs, full per-image
e2e-aware replay, CUDA kernel validated against the production PyTorch
implementation on real TensorRT outputs. No correctness or bookkeeping item
remains open. Temporal-window routing, UIoU, cold-reload replay, and the
numpy-overhead decomposition are **not required** to complete this paper.

**Artifact freeze — after this point, do not modify the policy code, risk
grid, data split, or metric definitions used below. A manuscript-wording-only
change never requires reopening this evidence package; a change to any of
the items below does.**

- **Commit**: this update ships in the same commit as this section (`git log
  -1` at the time this file was written; see the repo's commit history for
  the exact SHA — deliberately not hardcoded here per this project's
  existing convention of pointing to `git log` rather than a SHA that would
  go stale if this file is ever amended).
- **Canonical result files**: `reports/router_overhead_replay_E3_20260928
  .json` (seed0/E3), `reports/router_overhead_replay_E1_20260928.json`
  (seed0/E1), `reports/router_overhead_replay_E3_seed2_20260929.json`,
  `reports/router_overhead_replay_E1_seed2_20260929.json`,
  `reports/router_overhead_replay_E3_seed3_20260929.json`,
  `reports/router_overhead_replay_E1_seed3_20260929.json`.
- **Kernel audits**: `reports/audit_gpu_risk_kernel_E3.json` (2,000-trial
  synthetic arithmetic stress test), `reports/audit_gpu_risk_kernel_
  production_E3.json` (40-real-output integration audit).
- **Seed/checkpoint IDs**: `pace_seg_v1_seed0`, `pace_seg_v1_seed2`,
  `pace_seg_v1_aug_seed3` (all `step_00100000.pt`, 100k-step full runs).
- **Per-image dumps**: `reports/router_per_image_dump_seed0.json`,
  `_seed2.json`, `_seed3.json`.
- **Canonical risk-grid source per seed**: seed0 —
  `reports/router_E3_20260921.json` (verified byte-identical to
  `router_E1_20260921.json` for this seed); seed2 —
  `reports/router_E3_seed2_dump_meta.json`; seed3 —
  `reports/router_E3_seed3_dump_meta.json`. The same file is used for
  *both* E1 and E3's replay of that seed.
- **E1/E3 latency-table IDs**: `reports/router_overhead_E3_20260922.json`
  (TensorRT/CUDA, `gpu`-kernel entropy backend, warm/resident);
  `reports/router_overhead_E1_20260928.json` (Hailo-8, `numpy` entropy
  backend — the only one that exists on this architecture, warm/resident).
- **Replay command** (per seed/device, `--entropy-backend` selects which
  warm-latency table to read): `scripts/replay_router_with_overhead.py
  --per-image-dump reports/router_per_image_dump_seed{N}.json
  --overhead-json reports/router_overhead_{E1,E3}_2026....json
  --original-results-json <canonical grid file for that seed>
  --lookup-table outputs/benchmark_lookup_table.csv --device-id {E1,E3}
  --backend {hailo_hef,tensorrt_gpu} --entropy-backend {numpy,gpu}
  --output-json reports/router_overhead_replay_{device}_seed{N}_
  20260929.json`.
- **Kernel audit commands**: `scripts/audit_gpu_risk_kernel.py
  --engine-dir . --output-json reports/audit_gpu_risk_kernel_E3.json` (on
  E3); `scripts/capture_real_engine_outputs.py --engine-dir . --n-samples 10
  --output-dir real_outputs` (on E3) then `scripts/
  audit_gpu_risk_kernel_production.py --real-outputs-dir
  reports/real_outputs --output-json reports/audit_gpu_risk_kernel_
  production_E3.json` (on the dev machine, needs torch).
- **Final locked table**: the "Aggregated (Codex's requested summary
  format)" table above (E3 35/60, E1 34/60, Overall 69/120 fair — 51/16/2
  W/T/L, 97.1% win-or-tie, macro +0.0241 mIoU, D 0/120 violations, A 51/120
  violations) and the adopted claim/boundary/three-annotations block above
  this section are the frozen, final numbers for this closing requirement.

## Not yet done (all non-gating per Codex's explicit ruling — confirmed unnecessary to complete this paper)

- Cold/reload frontier replay (only warm/resident was used for the win/loss
  analysis above on either backend, per Codex's rule that cold is a
  stress-test/robustness number, not automatically gating the closing
  decision).
- Separating E3's naive-numpy path's ~40-80ms into (device-to-host transfer/
  sync) vs. (numpy softmax/entropy) vs. (calibrator/policy) components. (Not
  applicable to E1 the same way — E1's dominant cost, the activate/deactivate
  cycle, is already isolated as its own named mechanism.) If not done, must
  continue to call this only a "naive host-side risk-evaluation path" cost,
  never attribute the full 40-80ms to numpy entropy specifically.
- E2/E5 (also TensorRT/CUDA, same backend family as E3) were never in scope
  for this closing requirement per Codex's locked two-backend rule, and remain
  out of scope now that E1 confirms the same conclusion on a structurally
  different backend.
- Power measurement — no external calibrated meter on any device (locked as
  deferred future work).
