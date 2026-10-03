# Landscape rerun — post-training pipeline (prepared 2026-10-03)

Order of work once the nine landscape runs finish (see
`RESOLUTION_ORIENTATION_FIX_20261003.md`). Result files use the tag `20261004`.

## 1. Server — supernets (as soon as the three supernet checkpoints exist)

Run on one server (any GPU; the other queues may still be running):

```bash
cd ~/Dung_TDTU/imavis-edge-seg && source ~/miniconda3/bin/activate imavis-edge-seg && STAGE=supernets bash scripts/server/start_post_train_landscape.sh
```

Check: `tail -n 20 outputs/post_train_landscape_supernets.log` — finished at
`post-train supernets end`. Produces `reports/landscape_20261004/` (supernet eval with
per-class IoU, Run A/B/C router dumps = seeds 0/1/2, FLOPs) and the device bundle in
`~/Dung_TDTU/compiled_eval_landscape` (outside git).

## 2. Server — baselines (after Fast-SCNN ×3 and PACE-Large ×3)

```bash
cd ~/Dung_TDTU/imavis-edge-seg && source ~/miniconda3/bin/activate imavis-edge-seg && STAGE=baselines bash scripts/server/start_post_train_landscape.sh
```

## 3. Server — commit reports once, upload the bundle privately

```bash
cd ~/Dung_TDTU/imavis-edge-seg && git add reports/landscape_20261004 && git commit -m "report: landscape post-train evaluation" && git push origin main
source ~/miniconda3/bin/activate imavis-edge-seg && python -c "from huggingface_hub import HfApi; a=HfApi(); r=a.whoami()['name']+'/imavis-compiled-eval-landscape'; a.create_repo(r, repo_type='dataset', private=True, exist_ok=True); a.upload_folder(repo_id=r, repo_type='dataset', folder_path='/home/ubuntu/Dung_TDTU/compiled_eval_landscape'); print('UPLOAD_OK', r)"
```

The bundle contains Cityscapes/ACDC-derived images: keep the HF repo private and delete
it after transfer (dataset licences forbid redistribution).

## 4. Workstation — download bundle, edge measurements (Claude, via SSH/WSL)

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('DungJD/imavis-compiled-eval-landscape', repo_type='dataset', local_dir='tmp/compiled_eval_landscape')"
bash scripts/edge/landscape_devices.sh trt E3     # engines, LUT protocol, static-vs-route, compiled accuracy
bash scripts/edge/landscape_devices.sh trt E2     # engines + LUT protocol
bash scripts/edge/landscape_devices.sh trt E5
bash scripts/edge/landscape_devices.sh dfc        # Hailo HEFs with real INT8 calibration (WSL, DFC 3.34)
bash scripts/edge/landscape_devices.sh hailo      # E1: LUT protocol, static-vs-route (explicit/scheduler x f32/u8), compiled accuracy
bash scripts/edge/landscape_devices.sh lut        # outputs/benchmark_lookup_table_landscape.csv
```

Verify SHA-256 of the bundle against `summary.json` before use.

## 5. Workstation — analyses

```bash
bash scripts/run_landscape_analyses.sh
```

Runs A/A-hard/T-hard/D, best feasible static (direct same-harness cost), p95/p99, LOCO,
calibration quality, bootstrap and nested bootstrap CIs, break-even overhead, and the
on-device-prediction replay, all writing `reports/*_20261004.json`.

## 6. Figures and manuscript

Regenerate figures from the new JSONs (RQ1 figure from the landscape LUT; RQ2 from
`eval_supernet_seed*` vs `eval_fast_scnn_seed*` and `eval_pace_large_seed*`; qualitative
panels re-rendered in landscape), then write V20 around the agreed analysis framing.
