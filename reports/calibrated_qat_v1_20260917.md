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

## Update 2026-09-17, same day: real result -- calibration makes it *worse*, not
better

Ran the suggested command below (seed0, `large` level, same 10k-step/lr=3e-5
recipe as every other QAT run) and evaluated with `evaluate_supernet.py --qat`.

| split | FP32 | dynamic QAT (2026-09-16) | calibrated QAT | calibrated − dynamic |
|---|---|---|---|---|
| cityscapes | 0.5267 | 0.5107 (−1.60) | 0.5061 (**−2.06**) | −0.46 |
| acdc/fog | 0.5504 | 0.5492 (−0.12) | 0.5457 (**−0.47**) | −0.35 |
| acdc/night | 0.3728 | 0.3562 (−1.66) | 0.3507 (**−2.21**) | −0.55 |
| acdc/rain | 0.4983 | 0.4986 (+0.03) | 0.4851 (**−1.32**) | −1.35 |
| acdc/snow | 0.4852 | 0.4919 (+0.67) | 0.4802 (**−0.50**) | −1.17 |

**Calibrated ranges lose more than dynamic ranges on every single split** (0.35
to 1.35 points worse), the opposite of the leading hypothesis this feature was
built to test. This directly refutes "the supernet's dynamic range is the
problem, calibration will fix it" in its simplest form -- at least for this
specific calibration design (a single global max-observed-activation scale from
a 200-image calibration set).

A plausible explanation: max-based calibration is sensitive to outliers -- a
single unusually large activation anywhere in the ~200-image, 4-level
calibration pass sets the frozen scale for *every* later forward call,
including calls whose actual activations are much smaller and now get quantized
on a needlessly coarse grid. The dynamic mode's per-call max, by contrast, never
carries an outlier from one image into the quantization of a completely
different image. A percentile-based or running-average observer (rather than a
hard running max) is the natural next thing to try, not yet done.

**§11 QAT go bar on the supernet: still does not reliably pass, and the most
promising untested fix did not help.** The supernet-vs-independent-baseline QAT
gap documented in `reports/qat_v1_20260913.md`'s 3-seed dynamic-mode result
remains unexplained and unresolved.

## Not yet done

- A percentile/running-average calibration observer instead of a hard running
  max, to test whether outlier-sensitivity specifically is the calibrated
  mode's problem.
- Calibrating only some layers (e.g. just the layers nearest the input, where
  raw pixel-statistics outliers would show up most directly) rather than every
  layer uniformly.
- Only tested on seed0 so far -- a second calibrated-mode seed would confirm
  whether "calibration makes it worse" is a real, reproducible pattern or
  partly seed noise, the same caution already applied to the dynamic-mode
  result and the distillation ablation.
- Only the supernet's `SlimmableConv2d` path is exercised by
  `build_calibration_dataloader`'s wiring so far; the independent baselines'
  `QATConv2d` layers already pass the go bar under dynamic ranges, so calibrating
  them is lower priority, not yet done.

## Run command (for reference / re-running with a different seed or observer design)

```bash
cd ~/Dung_TDTU/imavis-edge-seg && git pull --ff-only && bash scripts/server/start_train_supernet.sh configs/experiment/default.yaml \
  --override experiment_id=pace_seg_v1_qat_calibrated_seed0 \
  --override seed=0 \
  --override training.max_steps=10000 \
  --override training.lr=3e-5 \
  --qat --calibrate \
  --init-checkpoint outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt
```
