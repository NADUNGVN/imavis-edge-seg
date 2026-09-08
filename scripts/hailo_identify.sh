#!/usr/bin/env bash
# Run ON device E1 (Pi5 + Hailo) once hailo-all/hailort is installed.
# Records the exact chip (Hailo-8 vs Hailo-8L), firmware, HailoRT, Model Zoo and DFC
# versions -- required before any Hailo latency/energy claim (RESEARCH_PLAN.md §14).
set -euo pipefail

OUT="${1:-hailo_identify_$(date +%Y%m%dT%H%M%S).txt}"

echo "== hailortcli fw-control identify ==" | tee "$OUT"
hailortcli fw-control identify | tee -a "$OUT"

echo "== hailortcli --version ==" | tee -a "$OUT"
hailortcli --version | tee -a "$OUT"

echo "== dpkg versions ==" | tee -a "$OUT"
dpkg -l | grep -i hailo | tee -a "$OUT"

echo "Wrote $OUT"
echo "Do not commit this file if it contains device serials/MACs; summarize into docs/INFRA_OVERRIDE.md instead."
