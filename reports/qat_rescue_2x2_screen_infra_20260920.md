# QAT rescue — 2×2 factorial screen infrastructure, 2026-09-20

Builds the infrastructure for the 2×2 factorial screen Codex proposed and both sides
agreed to (`docs/COORDINATION_LOG.md`'s "Standing rules"/open thread #1), to isolate
which of two independent factors — shared vs. independently-fine-tuned weights,
dynamic vs. calibrated activation range — actually matters for the supernet QAT gap
first documented in `reports/qat_v1_20260913.md` and left unresolved after
`reports/calibrated_qat_v1_20260917.md`'s max-observer calibration made things worse.
This report originally covered **infrastructure only**; see "Update: real 2×2
results" below for the completed screen, run same day on real hardware.

## The 4 cells

| | dynamic range | EMA/percentile-calibrated range |
|---|---|---|
| **shared supernet weights** | done, `reports/qat_v1_20260913.md` (3 seeds, worst case −2.54) | infra ready, not run |
| **exported/independent per-level weights** | infra ready, not run | infra ready, not run |

Deliberately not combining both axes in one attempt first — that would not
distinguish which factor (if either) helped. Run all 4 cells on seed0, then
replicate only the winning cell on a second seed.

## What was built

- **`training/quantization.py`**: a second calibration observer,
  `CalibrationObserver = Literal["max", "ema_percentile"]`. `"max"` is the original
  (`reports/calibrated_qat_v1_20260917.md`'s hard running maximum, confirmed
  outlier-sensitive and worse than dynamic). `"ema_percentile"` computes a per-call
  high percentile (default 99.9th, via `torch.topk` rather than `torch.quantile` —
  cheaper for a small excluded fraction on a large tensor, and avoids
  `torch.quantile`'s element-count ceiling on some PyTorch/CUDA versions) and
  combines it across calibration calls with an exponential moving average (default
  momentum 0.9), directly testing whether `"max"`'s single-outlier sensitivity was
  the actual problem. `run_calibration(model, forward_calls, observer="max",
  percentile=0.999, momentum=0.9)` — default unchanged, so every existing caller's
  behavior is bit-for-bit the same as before.
- **`training/baseline_trainer.py::run_baseline_training`**: new `model:
  nn.Module | None = None` parameter — when given, skips `build_baseline_model`
  entirely and fine-tunes the given module instead. Lets the exported-subnet script
  below reuse this same, already-well-tested training loop (optimizer/schedule/
  checkpoint/resume, all unchanged) rather than writing a second one. Also gained
  `calibrate`/`calibration_images`/`calibration_observer`/`calibration_percentile`/
  `calibration_momentum`/`calibration_level` parameters, mirroring
  `training/trainer.py`'s identical wiring but without the per-level loop (a
  baseline or exported-subnet model has exactly one resolution).
- **`training/data.py`**: `build_train_dataset`/`build_train_dataloader`/
  `build_calibration_dataloader` all gained an optional `level: ElasticityLevel |
  None = None` parameter (default `None` reproduces the previous "always load at
  the largest configured level's resolution" behavior exactly). Needed because an
  exported subnet only has one resolution, and it isn't always the largest level's.
- **`models/subnet.py`** (read, not modified): confirmed `extract_subnet` produces
  a genuinely independent `nn.Conv2d` graph (never `SlimmableConv2d`), so
  `apply_qat` converts it via the simpler, already-well-tested `QATConv2d` path —
  no new quantization class was needed. Also confirmed `StaticPaceSegSubnet.
  __init__` ends with `self.eval()`; the new script below explicitly re-`.train()`s
  the extracted subnet before fine-tuning, or BatchNorm running stats would never
  update.
- **`scripts/train_exported_subnet.py`** (new): loads an FP32 supernet checkpoint,
  calls `extract_subnet(supernet, level)`, `.train()`s the result, then calls the
  extended `run_baseline_training(model=<extracted>, qat=..., calibrate=...)`. Same
  `--qat`/`--calibrate`/`--calibration-*` flags as `train_baseline.py`/
  `train_supernet.py`.
- **`scripts/train_supernet.py`** / **`scripts/train_baseline.py`**: both gained
  `--calibration-observer {max,ema_percentile}`/`--calibration-percentile`/
  `--calibration-momentum` flags (previously hardcoded to `"max"`).
- **`scripts/server/{start,run,status}_train_exported_subnet.sh`** (new): detached
  server-job scripts mirroring `*_train_baseline.sh`'s pattern exactly (job dir
  keyed by elasticity level instead of baseline name, same state.env/log/
  `reports/server/*.md` conventions), so an exported-subnet run survives an SSH
  disconnect the same way every other long training job in this repo already does.
- 7 new tests in `tests/test_quantization.py` (percentile-via-topk correctness,
  outlier-ignoring behavior, `run_calibration` wiring for both observers, backward
  compatibility of the `"max"` default), 4 new tests in `tests/test_baselines.py`
  (`model=` override, calibration end-to-end against a real fixture, `calibrate`
  without `qat` raises, a real extracted-subnet fine-tunes-through-QAT integration
  test), 3 new tests in `tests/test_training_data.py` (`level=` overriding the
  loaded resolution for both `build_train_dataloader` and
  `build_calibration_dataloader`). **166/166 tests pass, ruff clean, mypy clean.**

## Not yet done

- All 4 cells still need a real server run — this is infrastructure only.
- The winning cell (if any) still needs a second-seed replication, per the agreed
  screening design.
- No new quantization class was needed for the exported-subnet path (confirmed via
  code reading, not just assumed), but this hasn't been cross-checked against a
  live server run yet either.

## Run commands (seed0, `large` level — cell 1 was already run at `large`, so these
match it for a clean comparison)

**Cell 2 — shared supernet × EMA/percentile observer:**

```bash
cd ~/Dung_TDTU/imavis-edge-seg && git pull --ff-only && bash scripts/server/start_train_supernet.sh configs/experiment/default.yaml \
  --override experiment_id=pace_seg_v1_qat_calibrated_ema_percentile_seed0 \
  --override seed=0 \
  --override training.max_steps=10000 \
  --override training.lr=3e-5 \
  --qat --calibrate --calibration-observer ema_percentile \
  --init-checkpoint outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt
```

**Cell 3 — exported subnet × dynamic range:**

```bash
cd ~/Dung_TDTU/imavis-edge-seg && git pull --ff-only && bash scripts/server/start_train_exported_subnet.sh large \
  outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt \
  configs/experiment/default.yaml \
  --override experiment_id=qat_exported_large_dynamic_seed0 \
  --override seed=0 \
  --override training.max_steps=10000 \
  --override training.lr=3e-5 \
  --qat
```

**Cell 4 — exported subnet × EMA/percentile observer:**

```bash
cd ~/Dung_TDTU/imavis-edge-seg && git pull --ff-only && bash scripts/server/start_train_exported_subnet.sh large \
  outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt \
  configs/experiment/default.yaml \
  --override experiment_id=qat_exported_large_ema_percentile_seed0 \
  --override seed=0 \
  --override training.max_steps=10000 \
  --override training.lr=3e-5 \
  --qat --calibrate --calibration-observer ema_percentile
```

Check progress with `bash scripts/server/status_train_supernet.sh` (cell 2) or
`bash scripts/server/status_train_exported_subnet.sh large` (cells 3/4). Once each
finishes, evaluate with `scripts/evaluate_supernet.py --qat` (cell 2, same as every
prior QAT evaluation) or:

```bash
uv run python scripts/evaluate_baseline.py --exported-subnet-level large --qat \
  --checkpoint outputs/qat_exported_large_dynamic_seed0/checkpoints/step_00010000.pt \
  --config configs/experiment/default.yaml
```

(cells 3/4 — `--exported-subnet-level`, new this session, rebuilds the matching
`extract_subnet` skeleton and evaluates at that level's own resolution rather than
the largest level's; caught and fixed live while writing this report, see below).

## Caught and fixed while writing this report

`scripts/evaluate_baseline.py` originally always called `build_baseline_model(name,
...)` to reconstruct the architecture before `load_state_dict`, with no way to
instead reconstruct an `extract_subnet` skeleton — an exported subnet's
`state_dict` keys/shapes don't match any registered `baseline_name`, so cells 3/4's
checkpoints would have had no way to be evaluated at all. Fixed by adding a
`--exported-subnet-level {tiny,small,medium,large}` flag (mutually exclusive with
`--model`) that builds `extract_subnet(PaceSegSupernet(config.supernet), level)` and
evaluates at that level's own configured resolution instead of the largest level's.
Verified with a new test
(`test_exported_subnet_qat_checkpoint_reloads_into_a_fresh_skeleton`): trains a QAT
exported subnet, saves a checkpoint, then reconstructs the same architecture from a
*differently-initialized* fresh supernet and confirms `load_state_dict` restores the
exact trained weights — the precise round trip `evaluate_baseline.py
--exported-subnet-level` depends on. 167/167 tests pass, ruff clean, mypy clean.

## Update, same day: real 2×2 results (seed0, `large` level)

All 4 cells ran on real hardware, evaluated with `evaluate_supernet.py --qat` (cell
1/2) or `evaluate_baseline.py --exported-subnet-level large --qat` (cell 3/4)
against the same FP32 reference checkpoint
(`outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt`, `large` level:
cityscapes 0.5267, acdc/fog 0.5504, acdc/night 0.3728, acdc/rain 0.4983, acdc/snow
0.4852). One real incident along the way: cell 3's first launch (SERVER-03) crashed
with a CUDA OOM before writing any checkpoint (another process held ~17GB on the
same GPU) — not a code issue, resolved by a plain retry once VRAM was free.

Also caught and fixed live: `scripts/server/{start,status}_train_exported_subnet.sh`
keyed their "latest run" pointer by level only, not by (level, hostname) — cells 3
and 4 both used `--level large` on different hosts, so the later launch silently
overwrote the earlier one's pointer, making cell 3 briefly unfindable via a bare
status check. Fixed the same way `status_train_supernet.sh` was fixed for the
identical bug class on 2026-09-16, plus an `<experiment_id>` search mode that
doesn't depend on the pointer file at all.

| cell | cityscapes | acdc/fog | acdc/night | acdc/rain | acdc/snow | worst-case |
|---|---|---|---|---|---|---|
| 1: shared × dynamic | −1.60 | −0.12 | −1.66 | +0.03 | +0.67 | **−1.66** |
| 2: shared × ema_percentile | −0.65 | −0.97 | −0.59 | −0.59 | −0.09 | **−0.97** |
| 3: exported × dynamic | −1.55 | +0.05 | −0.49 | −0.93 | +0.47 | **−1.55** |
| 4: exported × ema_percentile | −0.70 | −0.59 | −0.57 | +0.17 | +0.30 | **−0.70** |

(all values are mIoU points vs. the FP32 reference; positive = QAT *beats* FP32 on
that split)

**Against the §11 go bar (≤~1.0–1.5 points worst-case):** cells 2 and 4 pass
comfortably; cell 3 is right at the edge (−1.55, essentially tied with the bar);
cell 1 (the only previously-known cell) fails it, consistent with
`reports/qat_v1_20260913.md`'s 3-seed result.

**Attribution — holding one factor fixed at a time:**

- **Observer effect** (dynamic → ema_percentile), weights held fixed:
  shared: −1.66 → −0.97 (**+0.69** points); exported: −1.55 → −0.70 (**+0.85**
  points). Large, consistent improvement in both weight-sharing conditions.
- **Weight-sharing effect** (shared → exported), observer held fixed:
  dynamic: −1.66 → −1.55 (+0.11 points); ema_percentile: −0.97 → −0.70 (+0.27
  points). A real but much smaller improvement in both observer conditions.

**Preliminary reading (Codex owns the fuller observer-vs-weight-sharing analysis
and the go/stop recommendation per the agreed division of labor)**: the activation-
range observer is the dominant factor here, not shared-vs-independent weights. This
is consistent with — and stronger evidence for — the 2026-09-17 report's outlier-
sensitivity hypothesis about the `max` observer, and weakens the untested
"supernet's `large`-level weights are a harder quantization target because every
other elasticity level also exercises them" hypothesis as the primary explanation
(it may still contribute the smaller ~0.1–0.3 point weight-sharing effect observed
above, just not the dominant one).

**Best cell: 4 (exported subnet × ema_percentile), worst-case −0.70.** Per Codex's
pre-agreed decision rule (at least one cell reached the bar → confirm the best
config on a second seed), the next step is a seed1 confirmation run of cell 4 —
command below. No further observer/hyperparameter changes before that confirmation,
per the same agreement.

```bash
cd ~/Dung_TDTU/imavis-edge-seg && git pull --ff-only && bash scripts/server/start_train_exported_subnet.sh large \
  outputs/pace_seg_v1_aug_seed3/checkpoints/step_00100000.pt \
  configs/experiment/default.yaml \
  --override experiment_id=qat_exported_large_ema_percentile_seed3 \
  --override seed=3 \
  --override training.max_steps=10000 \
  --override training.lr=3e-5 \
  --qat --calibrate --calibration-observer ema_percentile
```

Note: uses `pace_seg_v1_aug_seed3` (seed=3) as the confirmation seed, not seed1 --
there is no `pace_seg_v1_aug_seed1` checkpoint. This repo's 3 existing supernet
seeds are `pace_seg_v1_aug_seed0`, `pace_seg_v1_aug_seed3`, `pace_seg_v1_seed2`
(`reports/qat_v1_20260913.md`); seed3 is also cell 1's own second seed
(worst-case −1.39 there), so this confirmation run doubles as a direct
apples-to-apples comparison against cell 1 on the same seed. A second-seed
confirmation must start from that seed's own FP32 weights, not reuse seed0's, or
it wouldn't really test seed-to-seed reproducibility.
