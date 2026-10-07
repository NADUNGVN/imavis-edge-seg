#!/usr/bin/env bash
# One-off (2026-10-07): PACE-Large baselines were trained with BatchNorm frozen in eval
# mode (StaticPaceSegSubnet.__init__ calls self.eval(); baseline_trainer never called
# model.train()). Move those checkpoints and their evaluations aside so the retrain
# queues and STAGE=baselines do not skip them. Run ONCE on any server (NFS is shared).
#   bash scripts/server/archive_frozen_bn_pace_large.sh
set -euo pipefail
cd "${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
A=outputs/_invalid_frozen_bn_20261007
mkdir -p "$A/reports"
for s in 0 1 2; do
  [ -d "outputs/pace_large_landscape_seed$s" ] && mv "outputs/pace_large_landscape_seed$s" "$A/"
  [ -f "reports/landscape_20261004/eval_pace_large_seed$s.json" ] && mv "reports/landscape_20261004/eval_pace_large_seed$s.json" "$A/reports/"
done
ls "$A" "$A/reports"
