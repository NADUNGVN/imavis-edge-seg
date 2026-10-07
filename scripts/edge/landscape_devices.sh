#!/usr/bin/env bash
# Edge-device measurements for the landscape rerun (2026-10-04). Run from the Windows
# workstation (Git Bash) in the repo root; uses the ssh aliases agx (E3), nx (E2),
# nano (E5), pi5 (E1) and WSL for the Hailo DFC. Each stage is idempotent-ish and leaves
# raw files on the device plus copies under reports/ or outputs/.
#
# Prereq: the server bundle (ONNX + held-out eval npz + calib npz) from
# scripts/server/post_train_landscape.sh STAGE=supernets, downloaded to $BUNDLE
# (default tmp/compiled_eval_landscape), e.g. via a PRIVATE HF dataset repo.
#
#   bash scripts/edge/landscape_devices.sh trt E3      # build engines, LUT protocol, static-vs-route, compiled eval
#   bash scripts/edge/landscape_devices.sh trt E2      # build engines + LUT protocol only
#   bash scripts/edge/landscape_devices.sh trt E5
#   bash scripts/edge/landscape_devices.sh dfc         # WSL: parse/optimize(real calib)/compile 4 HEFs
#   bash scripts/edge/landscape_devices.sh hailo       # E1: LUT protocol, static-vs-route (4 paths), compiled eval
#   bash scripts/edge/landscape_devices.sh lut         # build outputs/benchmark_lookup_table_landscape.csv
set -euo pipefail
TAG="${PACE_TAG:-20261004}"
BUNDLE="${BUNDLE:-tmp/compiled_eval_landscape}"
RAW=outputs/benchmark_raw_landscape
LEVELS="tiny small medium large"
declare -A HOST=([E3]=agx [E2]=nx [E5]=nano [E1]=pi5)
declare -A RDIR=([E3]='~/imavis_landscape' [E2]='~/imavis_landscape' [E5]='~/imavis_landscape' [E1]='~/pace_seg_landscape')

stage="${1:?stage required}"
case "$stage" in
trt)
  DEV="${2:?device E2|E3|E5}"; H=${HOST[$DEV]}; R=${RDIR[$DEV]}
  ssh "$H" "mkdir -p $R/onnx $R/engines $R/raw"
  scp -q "$BUNDLE"/onnx/*.onnx "$H:$R/onnx/"
  scp -q scripts/benchmark/run_trtexec_protocol.sh "$H:$R/"
  ssh "$H" "cd $R && for lv in $LEVELS; do [ -f engines/pace_seg_\$lv.engine ] || /usr/src/tensorrt/bin/trtexec --onnx=onnx/pace_seg_\$lv.onnx --saveEngine=engines/pace_seg_\$lv.engine --fp16 --buildOnly > engines/build_\$lv.log 2>&1; done; for lv in $LEVELS; do [ -d raw/\$lv ] || bash run_trtexec_protocol.sh onnx/pace_seg_\$lv.onnx $DEV gpu fp16 raw/\$lv > raw/\$lv.log 2>&1; done; ls engines raw"
  mkdir -p "$RAW/$DEV"; scp -q -r "$H:$R/raw/*" "$RAW/$DEV/"
  if [ "$DEV" = E3 ]; then
    scp -q scripts/measure_router_overhead.py "$H:$R/measure_router_overhead_v2lib.py"
    sed 's/^from measure_router_overhead import (/from measure_router_overhead_v2lib import (/' scripts/measure_static_vs_route_trt.py | ssh "$H" "cat > $R/measure_static_vs_route_trt.py"
    scp -q scripts/eval_compiled_engines.py "$H:$R/"
    ssh "$H" "mkdir -p $R/eval"; scp -q "$BUNDLE"/eval/*.npz "$H:$R/eval/"
    ssh "$H" "cd $R && export PATH=/usr/local/cuda/bin:\$PATH && ~/imavis_overhead/venv/bin/python -u measure_static_vs_route_trt.py --engine-dir engines --device-label E3 --output-mode both --output-json static_vs_route_E3_$TAG.json > static_vs_route.log 2>&1 && ~/imavis_overhead/venv/bin/python -u eval_compiled_engines.py --backend trt --engine-dir engines --eval-dir eval --output-prefix compiled_eval_E3_$TAG > compiled_eval.log 2>&1; tail -n 3 static_vs_route.log compiled_eval.log"
    scp -q "$H:$R/static_vs_route_E3_$TAG.json" "$H:$R/compiled_eval_E3_$TAG.json" "$H:$R/compiled_eval_E3_${TAG}_per_image.npz" reports/
  fi ;;
dfc)
  W=/mnt/d/Research/Teacher_Vu/IMAVIS_EDGE_SEG
  wsl.exe -e bash -lc "set -e; source ~/hailo-work/venvs/dfc-3.34/bin/activate; D=~/pace_seg_landscape_hailo; mkdir -p \$D/onnx \$D/calib \$D/logs; cp $W/$BUNDLE/onnx/*.onnx \$D/onnx/; cd \$D; python - <<'PY'
import numpy as np
mean=np.array([0.485,0.456,0.406],np.float32); std=np.array([0.229,0.224,0.225],np.float32)
for lv in ['tiny','small','medium','large']:
    u8=np.load('$W/$BUNDLE/calib/%s.npz'%lv)['images']; np.save('calib/%s.npy'%lv, ((u8.astype(np.float32)/255.0)-mean)/std)
PY
for lv in $LEVELS; do mkdir -p compiled_\$lv; [ -f compiled_\$lv/pace_seg_\$lv.hef ] && continue; hailo parser onnx onnx/pace_seg_\$lv.onnx --net-name pace_seg_\$lv --hw-arch hailo8 -y > logs/parse_\$lv.log 2>&1; hailo optimize pace_seg_\$lv.har --hw-arch hailo8 --calib-set-path calib/\$lv.npy --output-har-path pace_seg_\${lv}_optimized.har > logs/optimize_\$lv.log 2>&1; hailo compiler pace_seg_\${lv}_optimized.har --hw-arch hailo8 --output-dir compiled_\$lv > logs/compile_\$lv.log 2>&1; echo hef_ok \$lv; done"
  mkdir -p tmp/landscape_hef; for lv in $LEVELS; do wsl.exe -e bash -lc "cp ~/pace_seg_landscape_hailo/compiled_$lv/pace_seg_$lv.hef $W/tmp/landscape_hef/"; done; ls tmp/landscape_hef ;;
hailo)
  H=pi5; R=${RDIR[E1]}
  ssh "$H" "pgrep -af 'hailo|python' | grep -v -e pgrep -e hailort_service || true; mkdir -p $R/hef $R/eval $R/raw"
  scp -q tmp/landscape_hef/*.hef "$H:$R/hef/"
  scp -q scripts/benchmark/run_hailo_protocol.sh scripts/measure_router_overhead_hailo.py scripts/measure_static_vs_route_hailo.py scripts/eval_compiled_engines.py "$H:$R/"
  scp -q "$BUNDLE"/eval/*.npz "$H:$R/eval/"
  ssh "$H" "cd $R && for lv in $LEVELS; do [ -d raw/\$lv ] || bash run_hailo_protocol.sh hef/pace_seg_\$lv.hef E1 raw/\$lv > raw/\$lv.log 2>&1; done; for p in explicit scheduler; do for f in float32 uint8; do python3 -u measure_static_vs_route_hailo.py --hef-dir hef --path \$p --vstream-format \$f --output-json static_vs_route_E1_\${p}_\${f}_$TAG.json > svr_\${p}_\${f}.log 2>&1; done; done; python3 -u eval_compiled_engines.py --backend hailo --engine-dir hef --eval-dir eval --output-prefix compiled_eval_E1_$TAG > compiled_eval.log 2>&1; tail -2 compiled_eval.log"
  mkdir -p "$RAW/E1"; scp -q -r "$H:$R/raw/*" "$RAW/E1/"
  scp -q "$H:$R/static_vs_route_E1_*_$TAG.json" "$H:$R/compiled_eval_E1_$TAG.json" "$H:$R/compiled_eval_E1_${TAG}_per_image.npz" reports/ ;;
lut)
  .venv/Scripts/python.exe scripts/build_landscape_lut.py --raw-root "$RAW" --records-dir outputs/benchmark_records_landscape --output outputs/benchmark_lookup_table_landscape.csv ;;
*) echo "unknown stage $stage"; exit 1 ;;
esac
