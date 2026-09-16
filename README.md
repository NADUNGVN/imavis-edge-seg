# IMAVIS-EDGE-SEG

> Workspace path: `Teacher_Vu/IMAVIS_EDGE_SEG/` (one research track under the multi-paper
> `Teacher_Vu` workspace). Independent of `CARE_ASD/` — different modality (vision, not
> audio), different hardware profile, own git history.

**PACE-Seg: Platform-Aware Calibrated Elastic Semantic Segmentation for Reliable Edge
Vision under Adverse Conditions** (working title).

One elastic supernet trained once; 3-4 static INT8 subnets exported per accelerator
backend (TensorRT GPU, Xavier DLA, Hailo dataflow NPU); a calibrated visual-risk router
picks a subnet per frame window under a latency/energy budget measured on real hardware,
not estimated from FLOPs. Verified on Jetson Nano, Xavier NX, AGX Xavier and
Raspberry Pi 5 + Hailo-8 under clean, night, rain, fog and snow conditions.

Target venue: *Image and Vision Computing* (IMAVIS), Elsevier — special issue
**Complex Environment Vision**, deadline 2027-02-15.

> Full method design, research questions, datasets, baselines, metrics, ablations and
> go/no-go criteria live in [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) (adapted
> from the original research-gap analysis, kept as
> [`docs/SOURCE_RESEARCH_GAP_2026.md`](docs/SOURCE_RESEARCH_GAP_2026.md)).
>
> Shared server/hardware inventory follows
> [`../docs/SHARED_INFRASTRUCTURE.md`](../docs/SHARED_INFRASTRUCTURE.md); this project's
> deltas are in [`docs/INFRA_OVERRIDE.md`](docs/INFRA_OVERRIDE.md). Dataset access,
> layout and manifest workflow are in [`docs/DATASET.md`](docs/DATASET.md).
>
> Claude has no network path to `SERVER-01..05`; all server work goes through Git
> (push → one pasted command → committed report) per
> [`docs/COLLABORATION_PROTOCOL.md`](docs/COLLABORATION_PROTOCOL.md). Edge devices
> (`E1`/`E2`/`E3`/Jetson Nano) are SSHed into directly once reachable.

## Status

| Phase | Description | Status |
|------:|-------------|--------|
| 0 | Repository bootstrap (this scaffold) | **complete** |
| 1 | Hardware/toolchain inventory (Jetson Nano/NX/AGX, Hailo identify) | **E1 (Pi5+Hailo-8), E2 (Xavier NX), E3 (AGX Xavier), E5 (Orin Nano Super) all have working ML toolchains**, all verified 2026-09-08/09; E4 (RUBIK Pi 3) reachable, kept secondary. E5 is a newer Orin device, not the legacy EOL "Jetson Nano" the plan assumed — role vs. that board is an open decision. See `docs/INFRA_OVERRIDE.md` |
| 2 | Compiler smoke test (Fast-SCNN/BiSeNetV2 → TensorRT/DLA/HEF) | **complete — compile AND real-hardware-execution gate satisfied on all three backend families.** TensorRT GPU: 8/8 PASS on E2, E3 and E5. Xavier DLA (E2/E3): builds, but encoder-only — decoder falls back to GPU due to a confirmed hardware limit ("DLA supports only 16 subgraphs per DLA core", already fully consumed by the encoder alone), not an unsupported op. E5 has no DLA at all (confirmed). **DLA demoted to a secondary, encoder-only ablation 2026-09-10 (`RESEARCH_PLAN.md` §11 contingency executed) — TensorRT GPU + Hailo HEF are now the two primary backends** (`configs/experiment/default.yaml`). **Hailo: ONNX→HAR→HEF all 4 levels PASS, and all 4 HEFs ran successfully on real Hailo-8 hardware (E1)** 2026-09-10 (593.7/334.0/148.4/55.25 FPS, smoke-test numbers only, not a benchmark). See `scripts/compiler_smoke_test.md` and `reports/edge/` — includes a correction of an earlier wrong "0 fallback" claim. Calibrated INT8 not yet tried anywhere. |
| 3 | Benchmark harness + power measurement protocol | **core harness built and validated live on both TensorRT and Hailo hardware.** `src/imavis_edge_seg/benchmark/` implements the `docs/RESEARCH_PLAN.md` §9 protocol: latency stats + bootstrap 95% CI (`stats.py`), `trtexec --exportTimes` JSON and `hailortcli --csv` parsers (`parsers.py`), a `BenchmarkRecord` schema with JSON roundtrip and git-commit provenance (`report.py`), pooling of 3 independent runs per device into one record for both backends (`aggregate.py`), and a lookup-table aggregator that merges many records into one CSV for future Pareto search (`lookup_table.py`). `scripts/benchmark/run_trtexec_protocol.sh` **run live on E3 (AGX), all 4 elasticity levels x 3 runs, all PASS (2026-09-11)**; `scripts/benchmark/run_hailo_protocol.sh` **run live on E1 (Pi5) 2026-09-10, all 4 elasticity levels x 3 runs, all PASS**; **E2 (Xavier NX) and E5 (Orin Nano) added 2026-09-16** (`run_trtexec_protocol.sh`, all 4 levels x 3 runs each, all PASS, run directly via SSH from the dev machine with untrained-weight ONNX exports — `trtexec` latency doesn't depend on weight values) — see `reports/edge/E1_hailo_benchmark_protocol_20260910.md`, `reports/edge/E3_tensorrt_benchmark_all_levels_20260911.md`, `reports/edge/E2_E5_tensorrt_benchmark_all_levels_20260916.md`. **`outputs/benchmark_lookup_table.csv` now has a complete 4-device x 4-level latency table (16 rows)** — Hailo-8 (E1) + TensorRT GPU (E2/E3/E5), the two primary backends' full real-device coverage. **Found live: E1's Hailo-8 M.2 module has no on-board power/current sensor** (`--measure-power`/`--measure-current` both unsupported on this hardware) — energy/frame for E1 needs an external meter, not telemetry; the script and docs were corrected accordingly. 123/123 tests pass. **Not yet**: an external calibrated power meter for any device; INT8 precision latency anywhere; E4 (RUBIK Pi 3, Qualcomm QAIRT — a different toolchain, out of scope for this table under the current TensorRT+Hailo backend decision). |
| 4 | Elastic supernet v1 | **complete** — architecture, data pipeline (real Cityscapes 2975/500 + ACDC 1600/406), sandwich-rule + in-place-distillation + boundary-aware **training loop**, and an **mIoU evaluation loop**. 38/38 tests pass. **Model rescaled to ~1.02M params at "large"** (2026-09-10, comparable to Fast-SCNN; was ~126K) — re-ran the full compiler smoke test (TensorRT GPU/DLA on E2+E3, Hailo DFC compile + real E1 hardware run), 100% still PASS. **Same 2,000-step training budget at the new size: ~40-45% relative mIoU improvement at every level/condition over the old size** (Cityscapes large: 0.146→0.211) — direct empirical confirmation the capacity bottleneck was real, not just a prediction. Full detail: `reports/first_end_to_end_miou_20260910.md`, `reports/edge/model_rescale_compiler_revalidation_20260910.md`. **First real (100k-step) training run complete 2026-09-10/11** — Cityscapes large mIoU 0.211 (2k steps) → **0.4712** (100k steps), monotonic in model size at every level/condition holds at full budget too. Full detail: `reports/first_full_supernet_run_100k_20260910.md`. **Seed 1 also complete 2026-09-11** — within ~0.01-0.03 mIoU of seed 0 everywhere, 2/3 seeds done. **Per-class/ignore-pixel check on the acdc/fog > cityscapes-clean pattern: resolved, not a bug** — several structural classes score higher in fog, outweighing real degradation on dynamic road-user classes in the unweighted mean; reproduces identically in both seeds. **Data augmentation added 2026-09-11** (`data/transforms.py`'s `SegmentationTrainAugment`: random scale+crop, flip, color jitter; `TrainingConfig.augment`, default on) — only affects runs launched after this landed, not the ones already in flight. |
| 5 | Hardware-in-the-loop Pareto search | **v1 built and validated on real data 2026-09-12, extended to all 4 real devices 2026-09-16**: `src/imavis_edge_seg/search/pareto.py` joins real measured latency (`benchmark/lookup_table.py`) with real measured mIoU (`evaluate_supernet.py`), computes the Pareto frontier per (device, backend), and selects the best level under a latency budget. **Real run on all 4 benchmarked devices at the same 10ms budget: `small` on E1 (Hailo), `small` on E2 (Xavier NX), `large` on E3 (AGX Xavier), `medium` on E5 (Orin Nano)** — 3 different choices across 4 real devices at an identical budget, stronger evidence for RQ1 (hardware-aware selection matters) than the earlier 2-device result. All 4 elasticity levels remain Pareto-optimal on every device (no crossovers). Full detail: `reports/pareto_search_v1_20260912.md`, `reports/edge/E2_E5_tensorrt_benchmark_all_levels_20260916.md`. 123/123 tests pass. **Not yet**: a FLOPs-aware baseline to actually test RQ1's "≥15-20% better than FLOPs-aware" hypothesis (still the main gap); energy axis (no power meter yet); DLA latency data (demoted, encoder-only); wiring into the router (router currently uses only E1/E3, not yet re-run against E2/E5's now-available data). |
| 6 | QAT + distillation + compiler-safe refinement | In-place distillation implemented as part of Phase 4's sandwich-rule loop. **QAT v1 built 2026-09-13**: `src/imavis_edge_seg/training/quantization.py` — dynamic per-tensor INT8 fake-quantization, `QATConv2d`/`apply_qat` convert a model's `nn.Conv2d` layers in place while preserving `state_dict` key names exactly (needed for the FP32→QAT-finetune workflow, §5.2) — verified with a real `load_state_dict` round trip, and end to end against the real `fast_scnn` architecture (44/44 conv layers convert, forward/backward both work). **First real FP32-vs-INT8 result 2026-09-14**: `fast_scnn` fine-tuned 10k steps from its FP32 checkpoint with `--qat` (`scripts/train_baseline.py`), evaluated with `scripts/evaluate_baseline.py --qat` — loses 0.21-1.25 mIoU points vs. FP32 across cityscapes/fog/night/rain/snow (worst case acdc/rain: 0.5004→0.4879). **Confirmed 2026-09-15 across a second architecture (`ddrnet23_slim`, 5.2M params) and a second seed of `fast_scnn`** — both stay comfortably inside budget (`ddrnet23_slim` worst case -0.65 pts, even +0.13 on acdc/snow; `fast_scnn` seed1 worst case -1.04 pts). **§11 QAT go bar: passes, headline-ready — but only for independent baseline architectures.** **`QATSlimmableConv2d` added 2026-09-16** extends QAT to the actual supernet; the first real supernet result (`large` level, 2 seeds, 10k-step fine-tune) is **borderline, not a clean pass**: seed0 exceeds the go bar's budget on 2/5 splits (cityscapes -1.60, acdc/night -1.66), seed1 stays inside on all 5 (worst: acdc/fog -1.39) — worse than every independent-baseline result seen so far (worst case there: -1.25). A 3rd seed is needed before calling this confirmed either way; a plausible but untested explanation is that the supernet's shared weights (exercised by every elasticity level via the sandwich rule) may be a harder quantization target than an independently-trained model's. 123/123 tests pass. Full detail: `reports/qat_v1_20260913.md`. **Not yet**: a 3rd supernet QAT seed; calibrated (not dynamic) quantization ranges; `bisenetv2`/`segformer_b0`/`mobilenetv3_deeplabv3` not QAT-tested; supernet QAT only tested at `large`, not `tiny`/`small`/`medium`; compiler-safe refinement not started. |
| 7 | Calibrated visual-risk router | **v1 built 2026-09-14, run for real and extended 2026-09-15**: `src/imavis_edge_seg/router/` (`risk_probe`/`calibrator`/`policy`) + `scripts/evaluate_router.py` (fits a real `RiskCalibrator` on half of each val split, routes the held-out half per-image, measures each strategy's real achieved mIoU + real latency). First real run caught a genuine scale-mismatch bug (`entropy`'s raw nats-scale risk vs. `calibrated_risk`'s error-probability-scale risk compared against one shared threshold) — fixed same-day, then confirmed across **7/7 tested splits** (Cityscapes at 2 risk_target settings, E3/tensorrt_gpu + E1/hailo_hef, all 4 ACDC conditions): **`calibrated_risk` achieves higher achieved mIoU than `entropy` in every one** (margins +0.0013 to +0.0183), though it also spends a bit more latency each time — mIoU-per-ms efficiency is a mixed 2/7 vs. 5/7, so this is a real, direction-consistent RQ3 result, not yet a decisive efficiency win. **Structural finding**: `_select_by_risk`'s escalation only uses each candidate's latency *rank*, not magnitude — so a fitted calibrator's decisions port unchanged across deployment targets (E1 and E3 gave numerically identical achieved mIoU on the same split), a real robustness property, but the policy isn't yet latency-*value*-aware. 117/117 tests pass. Full detail: `reports/router_v1_20260914.md`. **Not yet**: latency-value-aware escalation; wiring into a streaming multi-engine loop (§5.3's 8-32 frame window, not per-image); a dedicated router-overhead-only measurement against the §11 "<~5%" bar; per-level (not just per-image) risk conditioning; UIoU (needs ACDC's uncertain-region annotations, not wired into the dataset loader). |
| 8 | Full Cityscapes/ACDC experiments | **Go/no-go: passes, headline-ready with 3 seeds per side (2026-09-14).** Full 3 augmented seeds each (`pace_seg_v1_seed2`/`aug_seed0`/`aug_seed3` vs. `fast_scnn_seed0_aug`/`seed1_aug`/`seed2_aug`, `RESEARCH_PLAN.md` §9 rule 8): supernet wins 3/5 splits, loses 2/5, every margin ≤1.1 mIoU points (near-exact tie on acdc/snow). Confirmed near-parity, not a win for either side — the honest claim is the shared supernet **matches** independent same-budget training within ~1 mIoU point at a fraction of the training/maintenance cost (RQ2's actual hypothesis, confirmed not exceeded). Comfortably clear of the §11 no-go bar either direction. Full detail: `reports/baseline_comparison_gap_check_20260912.md`. **Side finding on augmentation, completed 2026-09-16** (`reports/augmentation_effect_all_5_baselines_20260916.md`, correcting the 2026-09-15 2-data-point hypothesis in `reports/segformer_augmentation_investigation_20260915.md`): now that all 5 required baselines have an augmented run, the effect is **not** a monotonic function of model size. `fast_scnn` (1.1M params, worst no-aug fit of all 5 at 0.408 Cityscapes mIoU — catastrophically underfit) gains hugely from augmentation (+0.115 Cityscapes, +0.107 ACDC avg). `mobilenetv3_deeplabv3` (11.0M params, the *largest* model here) also gains a lot (+0.026 Cityscapes, +0.045 ACDC avg) despite not being underfit — its no-aug mIoU (0.577) is *worse* than `ddrnet23_slim`'s (0.627, less than half the params), consistent with overfitting the small (4575-image) training set instead, which augmentation's regularization then relieves. The 3 architectures in between (`bisenetv2` 1.7M, `segformer_b0` 3.7M, `ddrnet23_slim` 5.2M — the 3 *best* no-aug fits of the 5) show only small, mixed, sometimes slightly negative effects. Read as a bimodal/U-shaped relationship: augmentation helps most at *both* ends (underfit-relief for too-small models, overfit-relief for too-large ones), not "smaller models gain more." **All 5/7 required baseline slots now have both an augmented and non-augmented seed0 run** (`fast_scnn` additionally has 2 more augmented seeds for its 3-seed go/no-go); `pidnet_s` unneeded (DDRNet-23-slim satisfies "PIDNet-S or DDRNet-23-slim"); `hard`/`ucpnet` stay `NotImplementedError` (no public release found). **Fixed 2026-09-12**: checkpoint resume; a real DataLoader `persistent_workers` stall bug — the resume logic was directly exercised live 2026-09-16 when 2 of these 3 augmented runs were accidentally re-launched after already finishing; both correctly detected `step >= max_steps` and did nothing rather than re-train or overwrite. 117/117 tests pass. **Learned live**: shared-server free VRAM can change within the hour — always re-check `nvitop` immediately before launching. |
| 9 | Ablations + sustained thermal/power runs | not started |
| 10 | Manuscript | not started |

`SERVER-01..05` still have no direct SSH path (see `docs/COLLABORATION_PROTOCOL.md`).
The legacy Jetson Nano has not been provided (may not be needed — see
`docs/INFRA_OVERRIDE.md`). See
[`docs/INFRA_OVERRIDE.md`](docs/INFRA_OVERRIDE.md) for the full status.

## Quick start

### Requirements

- Python **3.11+**
- [uv](https://github.com/astral-sh/uv) (recommended)

### Install

```bash
uv sync --extra dev --extra torch --extra deploy
```

(`torch` is needed for the model code, `deploy` for the ONNX export test — both are
skipped gracefully if omitted.)

### Verify environment

```bash
uv run imavis-edge-seg --help
uv run imavis-edge-seg env-report
uv run imavis-edge-seg config-show --config configs/experiment/default.yaml
uv run pytest
uv run ruff check .
uv run mypy src
```

### Dataset manifests

Once Cityscapes/ACDC are downloaded somewhere (registration required — see
[`docs/DATASET.md`](docs/DATASET.md)):

```bash
uv run imavis-edge-seg data manifest --dataset cityscapes --data-root /path/to/cityscapes --split train -o data/manifests/cityscapes_train.csv
uv run imavis-edge-seg data manifest --dataset acdc --data-root /path/to/acdc --split train -o data/manifests/acdc_train.csv
```

### Training

```bash
uv run python scripts/train_supernet.py --config configs/experiment/default.yaml
```

On a server, prefer the detached wrapper so the job survives a closed SSH session (see
`docs/COLLABORATION_PROTOCOL.md`):

```bash
bash scripts/server/start_train_supernet.sh configs/experiment/default.yaml
bash scripts/server/status_train_supernet.sh
```

### Evaluation (mIoU)

```bash
uv run python scripts/evaluate_supernet.py --checkpoint outputs/<experiment_id>/checkpoints/step_XXXXXXXX.pt --config configs/experiment/default.yaml
```

Reports per-level mIoU on Cityscapes val and on ACDC val broken down by condition
(fog/night/rain/snow). Restrict with `--level`/`--dataset` (repeatable) for a faster
partial check.

## Project layout

```text
src/imavis_edge_seg/  # Library code (all logic lives here); models/ = supernet, data/ = datasets, training/ = sandwich-rule training loop, evaluation/ = mIoU
configs/               # YAML configs — supernet space, search, deployment, experiment
scripts/                # Thin CLI wrappers for long/server-side jobs; scripts/server/ = detached job wrappers
tests/                  # Unit / smoke tests
docs/                   # Research plan, infra override, hardware profile, protocols
data/                   # Manifests only in git; raw datasets gitignored
outputs/                # Generated (gitignored)
experiments/            # Registry + freeze files
reports/server/         # Small reviewable reports committed back from server tasks
```

## Research rules (summary)

- No claiming Hailo/DLA metrics until the corresponding backend is installed, smoke-tested
  and the run is reproducible on the real device — see `docs/INFRA_OVERRIDE.md`.
- FLOPs/parameter counts are never reported as a substitute for measured p95 latency and
  J/frame.
- Calibration and routing thresholds are fit on validation splits only, never on test
  labels (Cityscapes/ACDC/Dark Zurich).
- Every experiment records git commit, config hash, seed, and exact toolchain versions
  (JetPack/TensorRT/HailoRT/DFC/firmware).
- No overwriting prior experiment outputs — new `experiment_id` per run.

## License

MIT — see [`LICENSE`](LICENSE).
