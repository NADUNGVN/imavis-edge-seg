#!/bin/bash
# Run ON a Jetson device with trtexec available (see docs/INFRA_OVERRIDE.md for which
# devices have the ML stack installed). Builds every (elasticity level) x (GPU-FP16,
# GPU-INT8, DLA-FP16-no-fallback) combination for ONNX exports already copied to
# /tmp/pace_seg_<level>.onnx (produce those with export_subnet_onnx, see README.md).
#
# GPU-INT8 here has no calibration data -- it only proves the graph compiles in INT8
# mode, not that INT8 accuracy is acceptable. See scripts/compiler_smoke_test.md.
set -u
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
OUT=${OUT:-/tmp/smoke_matrix.log}
> "$OUT"

run() {
  local desc="$1"; shift
  echo "=== $desc ===" >> "$OUT"
  if "$TRTEXEC" "$@" >> "$OUT" 2>&1; then
    echo "RESULT: $desc PASS" >> "$OUT"
  else
    echo "RESULT: $desc FAIL" >> "$OUT"
  fi
  grep -E "Throughput:|Latency:|GPU Compute Time:|RESULT:" "$OUT" | tail -5
  echo "---"
}

for lvl in tiny small medium large; do
  onnx="/tmp/pace_seg_${lvl}.onnx"
  run "GPU-FP16-$lvl" --onnx="$onnx" --saveEngine="/tmp/eng_${lvl}_gpu_fp16.trt" --fp16
  run "GPU-INT8-$lvl" --onnx="$onnx" --saveEngine="/tmp/eng_${lvl}_gpu_int8.trt" --int8
  run "DLA-FP16-$lvl"  --onnx="$onnx" --saveEngine="/tmp/eng_${lvl}_dla_fp16.trt" --fp16 --useDLACore=0 --allowGPUFallback=false
done

echo "ALL_DONE" >> "$OUT"
