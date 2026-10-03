#!/usr/bin/env bash
# Post-training pipeline for the 2026-10-03 landscape retraining. Run on ONE server
# (conda env active) once the needed checkpoints exist; detached via nohup by
# start_post_train_landscape.sh. Never commits -- commit reports/landscape_20261004
# once by hand afterwards.
#
#   STAGE=supernets  -> needs the three supernet checkpoints:
#       evaluate_supernet (full val, per class)      eval_supernet_seed{0,1,2}.json   (RQ2)
#       evaluate_router  (fit/held-out dumps)        run_{a,b,c}_evaluation.json, run_{a,b,c}_per_image.json (RQ3)
#       prepare_compiled_eval for seed0              ONNX + held-out bundle in ~/Dung_TDTU/compiled_eval_landscape
#       compute_flops                                flops_landscape.json (RQ1)
#   STAGE=baselines  -> needs Fast-SCNN x3 and PACE-Large x3 checkpoints:
#       evaluate_baseline                            eval_{fast_scnn,pace_large}_seed{0,1,2}.json (RQ2)
# Each step is skipped if its output already exists, so the script can be rerun.
set -uo pipefail
cd "${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
STAGE="${STAGE:?STAGE=supernets or STAGE=baselines}"
OUT=reports/landscape_20261004
BUNDLE="${BUNDLE:-$HOME/Dung_TDTU/compiled_eval_landscape}"
CFG=configs/experiment/default.yaml
LUT=outputs/benchmark_lookup_table.csv   # only sets evaluate_router's legacy budget grid; replays use new measured costs
FINAL=step_00100000.pt
mkdir -p "$OUT"
echo "post-train $STAGE start $(date -u +%FT%TZ) host=$(hostname) git=$(git rev-parse --short HEAD)"

need() { [ -f "$1" ] || { echo "MISSING checkpoint $1 -- stage cannot run yet"; exit 2; }; }
step() {  # step <output> <command...>
  local out="$1"; shift
  if [ -e "$out" ]; then echo "skip $out"; return 0; fi
  echo "run  $out"; "$@" || { echo "FAILED $out"; exit 1; }
}

if [ "$STAGE" = supernets ]; then
  declare -A LABEL=([0]=a [1]=b [2]=c)
  for s in 0 1 2; do need "outputs/pace_seg_landscape_supernet_seed$s/checkpoints/$FINAL"; done
  for s in 0 1 2; do
    CK="outputs/pace_seg_landscape_supernet_seed$s/checkpoints/$FINAL"
    step "$OUT/eval_supernet_seed$s.json" python -u scripts/evaluate_supernet.py --checkpoint "$CK" --config "$CFG" --per-class --output-json "$OUT/eval_supernet_seed$s.json"
    L=${LABEL[$s]}
    step "$OUT/run_${L}_per_image.json" python -u scripts/evaluate_router.py --checkpoint "$CK" --config "$CFG" \
      --lookup-table "$LUT" --device-id E3 --backend tensorrt_gpu --device cuda --run-label "Run ${L^^} (landscape seed$s)" \
      --output-json "$OUT/run_${L}_evaluation.json" --dump-per-image "$OUT/run_${L}_per_image.json"
  done
  step "$BUNDLE/summary.json" python -u scripts/prepare_compiled_eval.py --checkpoint "outputs/pace_seg_landscape_supernet_seed0/checkpoints/$FINAL" --out-dir "$BUNDLE" --device cuda
  step "$OUT/flops_landscape.json" python -u scripts/compute_flops.py --baseline fast_scnn --baseline pace_large --output-json "$OUT/flops_landscape.json"
elif [ "$STAGE" = baselines ]; then
  for m in fast_scnn pace_large; do for s in 0 1 2; do need "outputs/${m}_landscape_seed$s/checkpoints/$FINAL"; done; done
  for m in fast_scnn pace_large; do
    for s in 0 1 2; do
      step "$OUT/eval_${m}_seed$s.json" python -u scripts/evaluate_baseline.py --model "$m" --checkpoint "outputs/${m}_landscape_seed$s/checkpoints/$FINAL" --config "$CFG" --per-class --output-json "$OUT/eval_${m}_seed$s.json"
    done
  done
else
  echo "unknown STAGE=$STAGE"; exit 1
fi
echo "post-train $STAGE end $(date -u +%FT%TZ)"
