#!/usr/bin/env bash
# Protocol-compliant latency/power/temp capture for one HEF on E1 (docs/RESEARCH_PLAN.md
# §9): 3 independent runs, >=5000 frames each, environment manifest. HailoRT's
# --measure-power/--measure-temp are on-chip telemetry, not an external calibrated
# meter (§9 rule 6) -- label results accordingly, never as a system-level number.
#
# NOTE: written 2026-09-10 but not yet run live -- E1's Hailo chip was busy with another
# researcher's job at the time (see docs/INFRA_OVERRIDE.md). Verify the flags still
# match `hailortcli run --help` before first use.
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
    --measure-power --measure-temp \
    -c 5000 --csv "$OUTDIR/run${run}.csv" \
    > "$OUTDIR/run${run}_log.txt" 2>&1
  echo "run $run exit=$?"
done
echo "DONE: $OUTDIR"
