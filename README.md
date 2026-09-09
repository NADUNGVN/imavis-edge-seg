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
| 2 | Compiler smoke test (Fast-SCNN/BiSeNetV2 → TensorRT/DLA/HEF) | **Compile-level gate satisfied on all three backend families.** TensorRT GPU: 8/8 PASS on E2, E3 and E5. Xavier DLA (E2/E3): builds, but encoder-only — decoder falls back to GPU due to a confirmed hardware limit ("DLA supports only 16 subgraphs per DLA core"), not an unsupported op. E5 has no DLA at all (confirmed). **Hailo DFC: ONNX→HAR→HEF all 4 levels PASS** (compiled on the researcher's WSL2, 2026-09-09) — not yet run on real Hailo-8 hardware (E1 currently unreachable). See `scripts/compiler_smoke_test.md` and `reports/edge/` — includes a correction of an earlier wrong "0 fallback" claim. Calibrated INT8 not yet tried anywhere. |
| 3 | Benchmark harness + power measurement protocol | not started |
| 4 | Elastic supernet v1 | **complete** — architecture (`src/imavis_edge_seg/models/`), data pipeline (`src/imavis_edge_seg/data/`, real Cityscapes 2975/500 + ACDC 1600/406 manifested on `SERVER-02`), and a working **sandwich-rule + in-place-distillation + boundary-aware training loop** (`src/imavis_edge_seg/training/`, `scripts/train_supernet.py`) — 32/32 tests pass including a synthetic end-to-end smoke run (loss decreases, checkpoint saves/loads, weights update). Detached server launch via `scripts/server/{start,status}_train_supernet.sh`. **Not yet**: a real training run on real data (only synthetic-data smoke-tested so far), any real/useful weights |
| 5 | Hardware-in-the-loop Pareto search | not started |
| 6 | QAT + distillation + compiler-safe refinement | in-place distillation implemented as part of Phase 4's sandwich-rule loop; QAT and compiler-safe refinement not started |
| 7 | Calibrated visual-risk router | not started |
| 8 | Full Cityscapes/ACDC experiments | not started — data + training loop ready, no real run launched yet |
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

## Project layout

```text
src/imavis_edge_seg/  # Library code (all logic lives here); models/ = supernet, data/ = datasets, training/ = sandwich-rule training loop
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
