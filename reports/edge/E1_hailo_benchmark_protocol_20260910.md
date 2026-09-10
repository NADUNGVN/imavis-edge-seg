# E1 (Pi5 + Hailo-8) — first live latency/energy benchmark run, 2026-09-10

Ran `scripts/benchmark/run_hailo_protocol.sh` (per `docs/RESEARCH_PLAN.md` §9: 3
independent runs, >=5000 frames, full environment disclosure) against all four
compiled `.hef` files from the rescaled (~1M-param) model
(`reports/edge/model_rescale_compiler_revalidation_20260910.md`), once E1's Hailo-8
chip was free of another researcher's concurrent job ("drone-rocket-deploy").

## Finding: this Hailo-8 module has no power/current sensor

The script originally passed `--measure-power`; the first live attempt failed:

```
[HailoRT CLI] [error] CHECK failed - Power measurement not supported. Disable the power-measure option
```

`--measure-current` fails the same way. `hailortcli fw-control identify` confirms
Board Name `Hailo-8` (not `Hailo-8L`), so this is not a module-family issue — the
specific M.2 card on E1 simply lacks the on-board power sensor HailoRT's
`--measure-power` assumes (unlike the PCIe eval boards HailoRT is normally
demonstrated against). `run_hailo_protocol.sh` was fixed to drop both flags and keep
`--measure-latency --measure-overall-latency --measure-temp` only. **Energy/frame for
E1 is unavailable from telemetry and requires an external calibrated power meter** —
per §9 rule 6, this must never be silently reported as 0 or omitted without
explanation; `BenchmarkRecord.power_mw` stays `None` for this device until one is used.

## Result: 4/4 levels, 3/3 runs each, all HAILO_SUCCESS

| Level | Resolution | end-to-end mean (ms) | kernel-only mean (ms) | derived FPS (1000/mean) | avg chip temp (C) |
|---|---|---:|---:|---:|---:|
| tiny | 384x192 | 3.714 | 1.946 | 269.3 | 45.9 |
| small | 512x256 | 6.597 | 4.642 | 151.6 | 48.1 |
| medium | 768x384 | 20.29 | 15.57 | 49.3 | 49.1 |
| large | 1024x512 | 41.14 | 32.71 | 24.3 | 50.1 |

Full stats (p50/p95/p99, bootstrap 95% CI, run-level samples) in
`outputs/benchmark_records/e1_hailo_<level>_fp16.json` and merged into
`outputs/benchmark_lookup_table.csv` (both gitignored under `outputs/`, regenerate via
`scripts/build_benchmark_lookup_table.py`).

**These derived-FPS numbers are lower than the earlier compiler-smoke-test FPS**
(593.7/334.0/148.4/55.25, recorded in
`reports/edge/model_rescale_compiler_revalidation_20260910.md`) because they measure
different things: the smoke test read `hailortcli`'s own streaming/pipelined
throughput counter (multiple frames in flight), while `1000/mean_latency_ms` here is
the stricter single-frame round-trip number, consistent with how
`build_record_from_trtexec_dir` already reports throughput for the TensorRT path.
Neither number is wrong; they answer different questions and must not be conflated.

## n=3 caveat for this backend

Unlike `trtexec --exportTimes` (thousands of per-iteration samples per run, pooled
across 3 runs into one CI), `hailortcli run --csv` reports one summary row per
invocation. `build_record_from_hailo_dir` (new in `benchmark/aggregate.py`) therefore
pools the 3 independent runs themselves into an n=3 sample — the bootstrap 95% CI on
that is a rough bound, not a tight one. This is a `hailortcli` output-format
limitation, not a measurement bug; documented in the function's docstring so a reader
of the JSON records doesn't mistake n=3 for a per-frame distribution.

## Consequence for the protocol script

`scripts/benchmark/run_hailo_protocol.sh` is now validated live on real hardware
across all 4 elasticity levels. `docs/INFRA_OVERRIDE.md` updated to record E1's
power-sensor limitation so a future run doesn't re-discover it from a failed
`hailortcli` invocation.
