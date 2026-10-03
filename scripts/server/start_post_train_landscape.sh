#!/usr/bin/env bash
# Detached launcher for post_train_landscape.sh (conda env active):
#   STAGE=supernets bash scripts/server/start_post_train_landscape.sh
#   STAGE=baselines bash scripts/server/start_post_train_landscape.sh
# Check: tail -n 20 outputs/post_train_landscape_<stage>.log
set -euo pipefail
cd "${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
STAGE="${STAGE:?STAGE=supernets or STAGE=baselines}"
if pgrep -f "[p]ost_train_landscape.sh" >/dev/null; then
  echo "a post-train pipeline is already running:"; pgrep -af "[p]ost_train_landscape.sh"; exit 1
fi
LOG="outputs/post_train_landscape_${STAGE}.log"
nohup setsid env STAGE="$STAGE" bash scripts/server/post_train_landscape.sh >"$LOG" 2>&1 </dev/null &
echo "post-train $STAGE started on $(hostname) pid=$! log=$LOG"
