#!/usr/bin/env bash
# Protocol-compliant latency capture for one (backend, precision) config on a Jetson
# device (docs/RESEARCH_PLAN.md §9): 3 independent runs, >=3s warm-up, exportTimes JSON
# per run (per-iteration h2d/compute/d2h/latency breakdown), environment manifest.
# This produces raw data only -- src/imavis_edge_seg/benchmark parses it afterwards.
#
# Usage: run_trtexec_protocol.sh <onnx_path> <device_id> <gpu|dla> <fp16|int8> <output_dir>
set -euo pipefail
ONNX="${1:?onnx path required}"
DEVICE_ID="${2:?device id required, e.g. E3}"
BACKEND="${3:?gpu or dla required}"
PRECISION="${4:?fp16 or int8 required}"
OUTDIR="${5:?output dir required}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$OUTDIR"

PREC_FLAG="--fp16"
[ "$PRECISION" = "int8" ] && PREC_FLAG="--int8"

DLA_FLAGS=""
[ "$BACKEND" = "dla" ] && DLA_FLAGS="--useDLACore=0 --allowGPUFallback=false"

{
  echo "device_id=$DEVICE_ID"
  echo "backend=$BACKEND"
  echo "precision=$PRECISION"
  echo "onnx=$ONNX"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$TRTEXEC" --help 2>&1 | grep -m1 -i 'tensorrt' || true
  nvpmodel -q 2>&1 || echo "nvpmodel: not available or needs sudo"
  cat /etc/nv_tegra_release 2>&1 || echo "no /etc/nv_tegra_release"
} > "$OUTDIR/environment.txt"

for run in 1 2 3; do
  echo "=== run $run/3 ($DEVICE_ID $BACKEND $PRECISION) ==="
  # shellcheck disable=SC2086
  "$TRTEXEC" --onnx="$ONNX" $PREC_FLAG $DLA_FLAGS \
    --warmUp=3000 --duration=15 --avgRuns=100 \
    --exportTimes="$OUTDIR/run${run}_times.json" \
    > "$OUTDIR/run${run}_log.txt" 2>&1
  echo "run $run exit=$?"
done
echo "DONE: $OUTDIR"
