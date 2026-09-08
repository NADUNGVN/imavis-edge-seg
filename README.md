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
> deltas are in [`docs/INFRA_OVERRIDE.md`](docs/INFRA_OVERRIDE.md).
>
> Claude has no network path to `SERVER-01..05`; all server work goes through Git
> (push → one pasted command → committed report) per
> [`docs/COLLABORATION_PROTOCOL.md`](docs/COLLABORATION_PROTOCOL.md). Edge devices
> (`E1`/`E2`/`E3`/Jetson Nano) are SSHed into directly once reachable.

## Status

| Phase | Description | Status |
|------:|-------------|--------|
| 0 | Repository bootstrap (this scaffold) | **complete** |
| 1 | Hardware/toolchain inventory (Jetson Nano/NX/AGX, Hailo identify) | **E1 (Pi5+Hailo-8) and E3 (AGX Xavier) have working ML toolchains** (HailoRT / CUDA+cuDNN+TensorRT, both verified 2026-09-08); E4 (RUBIK Pi 3) reachable, kept secondary; E2 (Xavier NX)/Jetson Nano still not provided; see `docs/INFRA_OVERRIDE.md` |
| 2 | Compiler smoke test (Fast-SCNN/BiSeNetV2 → TensorRT/DLA/HEF) | **TensorRT GPU + Xavier DLA: 12/12 PASS on E3** (2026-09-08, all 4 elasticity levels x FP16/INT8-uncalibrated/DLA-FP16, own architecture, random weights) — see `scripts/compiler_smoke_test.md` and `reports/edge/`. Hailo DFC (HEF) still untested — no Dataflow Compiler host set up. Xavier NX/Nano not yet tried. Calibrated INT8 not yet tried. |
| 3 | Benchmark harness + power measurement protocol | not started |
| 4 | Elastic supernet v1 | **architecture implemented** (`src/imavis_edge_seg/models/`) — slimmable-width + elastic-depth encoder-decoder, static subnet extraction verified numerically equal to the supernet, ONNX export tested; **not yet**: real training loop, dataset loading, sandwich-rule/distillation training, any real weights |
| 5 | Hardware-in-the-loop Pareto search | not started |
| 6 | QAT + distillation + compiler-safe refinement | not started |
| 7 | Calibrated visual-risk router | not started |
| 8 | Full Cityscapes/ACDC experiments | not started |
| 9 | Ablations + sustained thermal/power runs | not started |
| 10 | Manuscript | not started |

`SERVER-01..05` still have no direct SSH path (see `docs/COLLABORATION_PROTOCOL.md`).
E2 (Xavier NX) and Jetson Nano have not been provided yet. See
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

## Project layout

```text
src/imavis_edge_seg/  # Library code (all logic lives here); models/ = elastic supernet
configs/               # YAML configs — supernet space, search, deployment, experiment
scripts/                # Thin CLI wrappers for long/server-side jobs
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
