# Calibrated (not dynamic) QAT quantization ranges — v1, 2026-09-17

Closes the gap `reports/qat_v1_20260913.md` has flagged since QAT v1's introduction:
every prior QAT result used a *dynamic* activation quantization range (recomputed
from each forward call's own observed min/max) — a documented simplification, and
the leading untested hypothesis for why the shared supernet's QAT go-bar result was
mixed/does-not-reliably-pass while every independent baseline passed comfortably.
This builds the calibrated alternative and wires it end to end; it does not yet
include the real result of running it (needs a server — GPU + real Cityscapes/ACDC
data — neither available on this dev machine).

## What was built

- `src/imavis_edge_seg/training/quantization.py`: `QATConv2d`/`QATSlimmableConv2d`
  now support two activation-quantization modes. **Dynamic** (unchanged default):
  scale recomputed from the input's own min/max every forward call. **Calibrated**
  (new): a scale observed once from a real calibration pass and then frozen for
  every later forward call. Weight quantization is unchanged (still dynamic,
  since a weight tensor's own min/max is always exactly known — no data-dependent
  uncertainty the way an activation's range has, so no calibration is needed
  there).
  - `run_calibration(model, forward_calls)`: puts every QAT-converted layer into
    calibration mode, runs a caller-supplied sequence of forward passes (typically
    one call per (calibration image, elasticity level) pair, so every level's own
    active-channel slice gets its own calibrated range), then freezes the observed
    max into `calibrated_max` and flips `calibrated=True`.
  - Both `calibrated_max` and `calibrated` are registered as **persistent buffers**
    (part of `state_dict`), not plain Python attributes — a real correctness fix
    caught before shipping: a plain-attribute `calibrated` flag would reset to
    `False` every time a fresh model is constructed and a checkpoint reloaded
    (e.g. in `evaluate_supernet.py`), silently falling back to dynamic
    quantization even for a checkpoint that was actually trained under calibrated
    ranges — exactly the kind of silent train/eval mismatch this project has
    caught and corrected before (the QATConv2d state_dict-key bug, the router
    risk_target scale-mismatch bug). With both as persistent buffers,
    `evaluate_supernet.py --qat`'s existing build → apply_qat → load_state_dict
    flow needs **no changes at all** to correctly resume calibrated mode.
- `src/imavis_edge_seg/training/data.py`: `build_calibration_dataloader(config,
  max_images=200)` — an un-augmented, evenly-subsampled slice of the same
  Cityscapes+ACDC data `build_train_dataset` uses (already spans
  day/night/rain/fog/snow when both are configured, satisfying `RESEARCH_PLAN.md`
  §5.2's explicit calibration-set requirement, with no new data-loading code).
- `training/trainer.run_training` / `scripts/train_supernet.py`: new
  `--calibrate`/`--calibration-images` flags. Calibration runs once, right after
  `apply_qat` and before the resume-checkpoint check, so resuming an
  already-calibrated in-progress run correctly keeps its own calibration (loaded
  via `state_dict`) rather than re-calibrating mid-fine-tune. Raises if
  `--calibrate` is passed without `--qat`.
- 27 new tests across `test_quantization.py` (calibration mode dispatch, frozen
  vs. dynamic behavior, monotonic max growth, state_dict round-trip persistence,
  a real-supernet end-to-end check) and `test_training_data.py`/`test_training.py`
  (calibration dataloader construction and a full `run_training(qat=True,
  calibrate=True)` integration test against real fixture images). 150/150 tests
  pass, ruff clean, mypy clean.

## Not yet done

- **No real result yet.** This needs a server: fine-tune with `--qat --calibrate
  --init-checkpoint <fp32>`, evaluate with `evaluate_supernet.py --qat`, and
  compare against the existing dynamic-mode supernet QAT results (3 seeds,
  worst-per-seed loss -1.66/-1.39/-2.54 points, 2/3 exceeding the go bar) --
  the run command is below.
- Only the supernet's `SlimmableConv2d` path is exercised by
  `build_calibration_dataloader`'s wiring so far; the independent baselines'
  `QATConv2d` layers already pass the go bar under dynamic ranges, so calibrating
  them is lower priority, not yet done.
- `run_calibration`'s max-observed-value approach (not a percentile/histogram
  observer) is itself a simplification -- a single outlier pixel could set an
  overly conservative scale. Worth revisiting if the calibrated result doesn't
  close the gap.

## Suggested run (server, GPU + real data required)

```bash
cd ~/Dung_TDTU/imavis-edge-seg && git pull --ff-only && bash scripts/server/start_train_supernet.sh configs/experiment/default.yaml \
  --override experiment_id=pace_seg_v1_qat_calibrated_seed0 \
  --override seed=0 \
  --override training.max_steps=10000 \
  --override training.lr=3e-5 \
  --qat --calibrate \
  --init-checkpoint outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt
```

Then evaluate the same way every other supernet QAT checkpoint has been (`scripts/evaluate_supernet.py --qat --level large --per-class`) and compare against seed0's dynamic-mode result (`reports/qat_v1_20260913.md`: cityscapes -1.60, acdc/night -1.66) to see whether calibration closes those specific overshoots.
