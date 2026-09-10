# Model rescale (~126K -> ~1M params at "large") — compiler re-validation, 2026-09-10

Base channels (`src/imavis_edge_seg/models/channels.py`) scaled 3x
(`BASE_STEM_CHANNELS` 32→96, `BASE_STAGE_CHANNELS` (48,64,96)→(144,192,288)) so the
architecture is comparable in scale to the smallest required baseline
(`docs/RESEARCH_PLAN.md` §6.2, Fast-SCNN ~1.1M params) instead of the original
compiler-smoke-test-only sizing. Since the graph topology (number of layers, elasticity
structure) is unchanged and only channel widths changed, re-ran the full compiler smoke
test to confirm nothing regressed.

## New param counts

| Level | Params (old) | Params (new) |
|---|---:|---:|
| tiny | 5,483 | 32,723 |
| small | 20,611 | 152,035 |
| medium | 55,059 | 420,219 |
| large | 126,243 | **1,020,739** |

## Result: 12/12 TensorRT (E2+E3) PASS, 4/4 Hailo DFC PASS, 4/4 Hailo hardware PASS

Same matrix as `E2_NX_compiler_smoke_test_20260908.md` / `E3_compiler_smoke_test_20260908.md`,
re-run against the new ONNX exports on both devices (direct SSH, no server involved):

- TensorRT GPU FP16 + uncalibrated INT8: 8/8 PASS (E2 + E3, all 4 levels), no unsupported
  ops, no new errors.
- Xavier DLA: builds at every level on both devices. **DLA fallback layer-mention counts
  are essentially unchanged from the old (126K-param) model**: tiny 40 (was 41), small 52
  (was 52), medium 64 (was 64), large 88 (was 88). This confirms the earlier finding that
  the "16 subgraphs per DLA core" limit is a **graph-structure** constraint (number of
  partition boundaries in the decoder), not a channel-width one -- scaling width alone
  does not change which layers fall back.
- Hailo DFC (WSL, `~/pace_seg_hailo_test_v2/`): ONNX -> HAR -> optimized HAR -> HEF, 4/4
  PASS, no unsupported layers, no new warnings beyond the expected FP16-subnormal /
  no-calibrator notices already seen for the smaller model.
- **Hailo real hardware (E1):** all four new `.hef` files ran via `hailortcli run`:

| Level | FPS (old, ~126K) | FPS (new, ~1M) |
|---|---:|---:|
| tiny | 593.70 | 593.69 |
| small | 334.01 | 265.25 |
| medium | 148.39 | 63.73 |
| large | 55.25 | 31.56 |

FPS drops with the larger channel counts, as expected (more compute per layer); `tiny`
barely moved since its multiplier (0.25x) keeps its absolute channel counts small even
after the 3x base-channel scale.

## Reading

Scaling the architecture to a baseline-comparable size did not introduce any new
compiler-compatibility problem on any of the three backend families. The known DLA
limitation (encoder-only) is unaffected by this change and remains open (see
`E3_compiler_smoke_test_20260908.md`).

## Consequence for existing checkpoints

Any checkpoint trained before this change (e.g. `outputs/pace_seg_dev_smoke/` on
`SERVER-02`, see `reports/first_end_to_end_miou_20260910.md`) has a **different
`state_dict` shape** now -- `PaceSegSupernet.load_state_dict` will fail against it. A
fresh training run is required; the old checkpoint/report remains valid as a pipeline
correctness record, not as a starting point for further training.
