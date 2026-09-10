#!/usr/bin/env bash
# Protocol-compliant latency/temp capture for one HEF on E1 (docs/RESEARCH_PLAN.md §9):
# 3 independent runs, >=5000 frames each, environment manifest. HailoRT's
# --measure-temp is on-chip telemetry, not an external calibrated meter (§9 rule 6) --
# label results accordingly, never as a system-level number.
#
# NOTE: --measure-power/--measure-current are deliberately NOT used. Confirmed live on
# E1 2026-09-10: this Hailo-8 M.2 module has no on-board power/current sensor --
# `hailortcli run --measure-power` fails with "Power measurement not supported" (and
# --measure-current likewise), unlike the PCIe eval boards HailoRT assumes. Energy/frame
# for E1 is unavailable from telemetry and needs an external meter -- do not report
# power_mw for this device without one.
#
# Usage: run_hailo_protocol.sh <hef_path> <device_id> <output_dir>
set -euo pipefail
HEF="${1:?hef path required}"
DEVICE_ID="${2:?device id required, e.g. E1}"
OUTDIR="${3:?output dir required}"
mkdir -p "$OUTDIR"

{
  echo "device_id=$DEVICE_ID"
  echo "hef=$HEF"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  hailortcli --version 2>&1 || true
  hailortcli fw-control identify 2>&1 || true
} > "$OUTDIR/environment.txt"

for run in 1 2 3; do
  echo "=== run $run/3 ($DEVICE_ID) ==="
  hailortcli run "$HEF" --measure-latency --measure-overall-latency \
    --measure-temp --dont-show-progress \
    -c 5000 --csv "$OUTDIR/run${run}.csv" \
    > "$OUTDIR/run${run}_log.txt" 2>&1
  echo "run $run exit=$?"
done
echo "DONE: $OUTDIR"
