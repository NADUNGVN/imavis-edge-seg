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
| 3 | Benchmark harness + power measurement protocol | **core harness built and validated live on both TensorRT and Hailo hardware.** `src/imavis_edge_seg/benchmark/` implements the `docs/RESEARCH_PLAN.md` §9 protocol: latency stats + bootstrap 95% CI (`stats.py`), `trtexec --exportTimes` JSON and `hailortcli --csv` parsers (`parsers.py`), a `BenchmarkRecord` schema with JSON roundtrip and git-commit provenance (`report.py`), pooling of 3 independent runs per device into one record for both backends (`aggregate.py`), and a lookup-table aggregator that merges many records into one CSV for future Pareto search (`lookup_table.py`). `scripts/benchmark/run_trtexec_protocol.sh` **run live on E3 (AGX) 2026-09-10**; `scripts/benchmark/run_hailo_protocol.sh` **run live on E1 (Pi5) 2026-09-10, all 4 elasticity levels x 3 runs, all PASS** — see `reports/edge/E1_hailo_benchmark_protocol_20260910.md`. **Found live: E1's Hailo-8 M.2 module has no on-board power/current sensor** (`--measure-power`/`--measure-current` both unsupported on this hardware) — energy/frame for E1 needs an external meter, not telemetry; the script and docs were corrected accordingly. 51/51 tests pass. **Not yet**: an external calibrated power meter for any device; TensorRT-side records covering all 4 levels (only `tiny` done so far); the full 5-device sweep. |
| 4 | Elastic supernet v1 | **complete** — architecture, data pipeline (real Cityscapes 2975/500 + ACDC 1600/406), sandwich-rule + in-place-distillation + boundary-aware **training loop**, and an **mIoU evaluation loop**. 38/38 tests pass. **Model rescaled to ~1.02M params at "large"** (2026-09-10, comparable to Fast-SCNN; was ~126K) — re-ran the full compiler smoke test (TensorRT GPU/DLA on E2+E3, Hailo DFC compile + real E1 hardware run), 100% still PASS. **Same 2,000-step training budget at the new size: ~40-45% relative mIoU improvement at every level/condition over the old size** (Cityscapes large: 0.146→0.211) — direct empirical confirmation the capacity bottleneck was real, not just a prediction. Full detail: `reports/first_end_to_end_miou_20260910.md`, `reports/edge/model_rescale_compiler_revalidation_20260910.md`. **First real (100k-step) training run complete 2026-09-10/11** — Cityscapes large mIoU 0.211 (2k steps) → **0.4712** (100k steps), monotonic in model size at every level/condition holds at full budget too. Full detail: `reports/first_full_supernet_run_100k_20260910.md`. **Not yet**: data augmentation (currently none, anywhere in the pipeline — likely the highest-leverage next change); a per-class/ignore-pixel check on the surprising acdc/fog > cityscapes-clean mIoU pattern; more than 1 training seed |
| 5 | Hardware-in-the-loop Pareto search | not started |
| 6 | QAT + distillation + compiler-safe refinement | in-place distillation implemented as part of Phase 4's sandwich-rule loop; QAT and compiler-safe refinement not started |
| 7 | Calibrated visual-risk router | not started |
| 8 | Full Cityscapes/ACDC experiments | 1x 100k-step supernet run **complete** on SERVER-02 (`pace_seg_v1`, `reports/first_full_supernet_run_100k_20260910.md`) — first real mIoU numbers, not yet compared against any baseline. **Scale-out infra ready**: a second supernet seed can be launched on an idle server via `scripts/server/start_train_supernet.sh <config> -- --override seed=1 experiment_id=pace_seg_v1_seed1` with zero new code. **Required-baseline training infra built** (`docs/RESEARCH_PLAN.md` §7): `scripts/train_baseline.py` / `scripts/server/{start,run,status}_train_baseline.sh` reuse the same data pipeline, loss and checkpoint format as the supernet trainer. Only 1/7 required baselines implemented so far (`mobilenetv3_deeplabv3`, via `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large` — needs no custom architecture code); the other 6 (Fast-SCNN, BiSeNetV2, PIDNet-S/DDRNet-23-slim, SegFormer-B0, HARD, UCPNet) raise an explicit `NotImplementedError` from `models/baselines.py` and still need architecture code. 60/60 tests pass. Ready to launch on SERVER-01/03/04/05 once confirmed idle. |
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
