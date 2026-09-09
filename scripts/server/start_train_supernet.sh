#!/usr/bin/env bash
# Start a detached supernet training run. Activate the right conda env first, then:
#   bash scripts/server/start_train_supernet.sh [config.yaml] [-- extra train_supernet.py args]
# Check progress with status_train_supernet.sh. Never paste a full training loop
# directly into an interactive shell -- this owns nohup/setsid so the SSH session can
# close without killing the job.
set -euo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"

if [ -n "$(git status --porcelain)" ]; then
  printf 'Refusing to start: server worktree is not clean.\n'
  git status --short
  exit 1
fi
if pgrep -af '[r]un_train_supernet.sh' >/dev/null; then
  printf 'A training run is already active.\n'
  exit 1
fi

CONFIG="${1:-configs/experiment/default.yaml}"
shift || true
RUN_ID="$(hostname)_train_$(date -u +%Y%m%dT%H%M%SZ)"
JOB_DIR="outputs/train_supernet/$RUN_ID"
mkdir -p "$JOB_DIR"
printf '%s\n' "$RUN_ID" > outputs/train_supernet/latest_run_id.txt

nohup setsid bash scripts/server/run_train_supernet.sh "$RUN_ID" "$CONFIG" "$@" \
  >"$JOB_DIR/launcher.log" 2>&1 </dev/null &
printf 'Training started: run_id=%s pid=%s\n' "$RUN_ID" "$!"
