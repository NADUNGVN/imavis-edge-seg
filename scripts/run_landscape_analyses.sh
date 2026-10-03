#!/usr/bin/env bash
# Run every router analysis on the landscape rerun (2026-10-04) from the Windows
# workstation once these exist locally (git pull for the reports/ JSONs):
#   reports/landscape_20261004/run_{a,b,c}_{evaluation,per_image}.json   (server, STAGE=supernets)
#   reports/static_vs_route_E3_<TAG>.json, static_vs_route_E1_*_<TAG>.json (edge, landscape_devices.sh)
#   reports/compiled_eval_E{1,3}_<TAG>[_per_image.npz]                      (edge)
#   outputs/benchmark_lookup_table_landscape.csv                             (edge, stage lut)
# Route costs come from the same-harness measurements (no separate overhead run).
#   bash scripts/run_landscape_analyses.sh
set -euo pipefail
export PACE_TAG="${PACE_TAG:-20261004}"
export PACE_ROUTER_BASE="${PACE_ROUTER_BASE:-reports/landscape_20261004}"
export PACE_ROUTE_COSTS=same_harness
export PACE_LUT="${PACE_LUT:-outputs/benchmark_lookup_table_landscape.csv}"
export PYTHONPATH=scripts
PY="${PY:-.venv/Scripts/python.exe}"
for f in "$PACE_ROUTER_BASE"/run_{a,b,c}_per_image.json reports/static_vs_route_E3_$PACE_TAG.json \
         reports/static_vs_route_E1_explicit_float32_$PACE_TAG.json "$PACE_LUT"; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
run() { echo "== $*"; "$PY" "$@" 2>&1 | grep -v Warning | tail -n 25; }
run scripts/router_review_analyses.py --bootstrap-reps 1000      # A, A-hard, D, static(route), p95/p99, LOCO, calibration
run scripts/router_review_analyses_v2.py                         # T-hard, static with LUT cost
run scripts/router_same_harness_analysis.py                      # static vs route, same harness (all E1 paths)
run scripts/router_breakeven.py                                  # break-even overhead
run scripts/router_nested_bootstrap.py --reps 200                # fit+held-out resampling
if [ -f "reports/compiled_eval_E3_${PACE_TAG}_per_image.npz" ] && [ -f "reports/compiled_eval_E1_${PACE_TAG}_per_image.npz" ]; then
  run scripts/router_ondevice_replay.py                          # on-device predictions (Run A)
fi
echo "done: reports/*_${PACE_TAG}.json"
