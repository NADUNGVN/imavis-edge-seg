# Input-orientation fix and landscape retraining (2026-10-03)

## What was wrong

`SupernetConfig.input_resolutions` (bootstrap commit 586e056, 2026-09-08) stored
`tiny=(384, 192) … large=(1024, 512)` without a stated convention. Every consumer
unpacks `height, width = input_resolutions[level]`, so all V1 models were trained,
evaluated, exported (ONNX `1x3x384x192`), compiled and benchmarked on **portrait,
aspect-distorted** inputs, while the manuscript describes 384×192 as width×height.
Confirmed independently by Claude Code and Codex; no document records it as intended.

## Fix

- `config.py`: tuples are now `(height, width)` and landscape: tiny (192, 384), small
  (256, 512), medium (384, 768), large (512, 1024). A validator rejects any portrait
  entry. `tests/test_orientation.py` guards the defaults and the eval transform shape.
- Pixel counts per level are unchanged, so FLOPs and memory are expected to be similar.
- Every V1 checkpoint, prediction cache, engine, HEF, FLOPs and latency number belongs to
  the old portrait regime and must not be mixed with landscape results.

## Retraining batch (9 runs × 100k steps)

| Server | GPU | Queue (sequential) |
|---|---|---|
| SERVER-01 | Quadro RTX 8000 48 GB | supernet seed0 → PACE-Large seed0 → Fast-SCNN seed0 |
| SERVER-02 | Quadro RTX 8000 48 GB | supernet seed1 → Fast-SCNN seed1 |
| SERVER-03 | RTX 3090 24 GB | supernet seed2 → PACE-Large seed1 → PACE-Large seed2 → Fast-SCNN seed2 |

Queue files: `configs/queues/landscape_server0{1,2,3}.txt`. PACE-Large standalone
(`build_baseline_model("pace_large")`) is the exact large static graph of a randomly
initialized supernet, trained alone with the baseline recipe (no sharing/distillation).
The queue assignment may be rebalanced after the throughput benchmark.

Recipe: unchanged from V1 (AdamW 3e-4, wd 1e-4, cosine, 500 warm-up, batch 8, clip 5.0,
boundary weight 1, α_distill 0.5, T 1, same augmentation), plus speed settings that are
identical for all nine runs: `training.amp=true` (FP16 autocast + GradScaler),
`training.cudnn_benchmark=true`, `training.num_workers` from the benchmark.

## Procedure

1. `scripts/benchmark_train_speed.py` on all three servers (concurrently, like the real
   batch) → pick workers/AMP and rebalance queues.
2. `SMOKE=1 bash scripts/server/start_queue.sh <queue>` (30 steps per job) on each server.
3. Full run: `bash scripts/server/start_queue.sh <queue>`; monitor with
   `bash scripts/server/status_queue.sh`.
4. Do not `git pull` or edit the shared NFS worktree while the batch runs. Jobs never
   commit; reports are committed once after the batch.
