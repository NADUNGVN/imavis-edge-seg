# QAT rescue — 2×2 factorial screen infrastructure, 2026-09-20

Builds the infrastructure for the 2×2 factorial screen Codex proposed and both sides
agreed to (`docs/COORDINATION_LOG.md`'s "Standing rules"/open thread #1), to isolate
which of two independent factors — shared vs. independently-fine-tuned weights,
dynamic vs. calibrated activation range — actually matters for the supernet QAT gap
first documented in `reports/qat_v1_20260913.md` and left unresolved after
`reports/calibrated_qat_v1_20260917.md`'s max-observer calibration made things worse.
This report covers **infrastructure only** — no server (GPU + real Cityscapes/ACDC
data) is available on this dev machine, so none of the 4 cells has a real result yet.

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
