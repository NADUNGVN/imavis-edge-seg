# E3 (AGX Xavier) — TensorRT GPU/FP16 latency benchmark, all 4 levels, 2026-09-11

Extended `scripts/benchmark/run_trtexec_protocol.sh` from tiny-only (the first live
validation, `reports/first_full_supernet_run_100k_20260910.md`'s benchmark-harness
predecessor work) to all 4 elasticity levels, using the same rescaled (~1.02M-param)
ONNX exports the Hailo/E1 benchmark used. Done independently of the train-server work
(direct SSH access to E3, no contention with SERVER-01/02/03's training jobs) while
baseline training runs were in flight.

## Result: 4/4 levels, 3/3 runs each, trtexec exit 0 throughout

| Level | Resolution | end-to-end mean (ms) | kernel-only mean (ms) | derived FPS (1000/mean) |
|---|---|---:|---:|---:|
| tiny | 384x192 | 0.912 | 0.682 | 1096.1 |
| small | 512x256 | 1.781 | 1.330 | 561.4 |
| medium | 768x384 | 4.546 | 3.514 | 220.0 |
| large | 1024x512 | 9.389 | 7.490 | 106.5 |

Full stats (p50/p95/p99, bootstrap 95% CI) in
`outputs/benchmark_records/e3_tensorrt_<level>_fp16.json` (gitignored, regenerate via
`build_record_from_trtexec_dir`). TensorRT version confirmed `8502` from the live
banner at every level, consistent with the earlier tiny-only run.

## Now have a complete 2-backend x 4-level latency table

`outputs/benchmark_lookup_table.csv` (regenerate via
`scripts/build_benchmark_lookup_table.py`) now has 8 rows: E1 (Hailo HEF) x 4 levels +
E3 (TensorRT GPU FP16) x 4 levels -- the two backends `RESEARCH_PLAN.md` §11's DLA
contingency designated primary. This is the first complete cross-backend latency
dataset and the direct input `docs/RESEARCH_PLAN.md` §5.3 B's Pareto-search lookup
table needs.

## Fixed a local tooling gap while building this

`build_lookup_table()` recursively globs `*.json` under `outputs/benchmark_records/` --
placing the raw `run*_times.json` scratch dumps in a subdirectory of that same tree (as
`trtexec_e3_raw/` had been) makes it try to parse trtexec's raw per-iteration JSON as a
`BenchmarkRecord` and crash. Raw per-device dumps now live under
`outputs/benchmark_raw/<name>/` instead, sibling to (not nested inside)
`benchmark_records/` -- convention only, no code change (both are gitignored).

## Not yet done

- TensorRT on E2 (Xavier NX) and E5 (Orin Nano Super) -- only E3 benchmarked so far.
- INT8 precision anywhere (only FP16 done).
- Xavier DLA benchmark (demoted to secondary ablation, not blocking, but not measured
  either).
