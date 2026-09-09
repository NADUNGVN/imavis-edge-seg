#!/usr/bin/env bash
# Read-only status check for the current/last supernet training run. No polling sleep
# -- run this once, read the answer, decide what's next.
set -uo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"
RUN_FILE="outputs/train_supernet/latest_run_id.txt"

if [ ! -f "$RUN_FILE" ]; then
  printf 'No training run has been started.\n'
  exit 0
fi

RUN_ID="$(cat "$RUN_FILE")"
JOB_DIR="outputs/train_supernet/$RUN_ID"
printf 'RUN_ID=%s\n\nSTATE:\n' "$RUN_ID"
[ -f "$JOB_DIR/state.env" ] && cat "$JOB_DIR/state.env" || printf 'state file not created yet\n'
printf '\nPROCESS:\n'
pgrep -af '[r]un_train_supernet.sh|[t]rain_supernet.py' || printf 'no matching process\n'
printf '\nLOG TAIL:\n'
[ -f "$JOB_DIR/train.log" ] && tail -n 30 "$JOB_DIR/train.log" || printf 'no log yet\n'
