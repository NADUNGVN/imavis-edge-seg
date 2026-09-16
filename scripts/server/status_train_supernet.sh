#!/usr/bin/env bash
# Read-only status check for a supernet training run. No polling sleep -- run this
# once, read the answer, decide what's next.
#
#   bash scripts/server/status_train_supernet.sh                   # this host's own latest run
#   bash scripts/server/status_train_supernet.sh <experiment_id>   # find a specific experiment_id's run, any host
#
# outputs/ is a *shared* NFS mount across SERVER-01..05 (docs/INFRA_OVERRIDE.md).
# With no argument, this reports the CURRENT HOST's own latest launch (tracked via a
# per-hostname pointer file), not whichever server anywhere launched most recently --
# a 2026-09-16 incident ran 3 different supernet experiments on 3 different servers
# concurrently, and every server's bare status check reported the *same* (most
# recently launched, on a different server) run, because the old version read one
# global outputs/train_supernet/latest_run_id.txt shared by every server. Pass the
# experiment_id explicitly to check a *specific* run regardless of which host it was
# launched from.
set -uo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"

EXPERIMENT_ID="${1:-}"

if [ -n "$EXPERIMENT_ID" ]; then
  RUN_ID=""
  for log in $(ls -t outputs/train_supernet/*/train.log 2>/dev/null); do
    if grep -qE "experiment_id=${EXPERIMENT_ID}( |\$)" "$log" 2>/dev/null; then
      RUN_ID="$(basename "$(dirname "$log")")"
      break
    fi
  done
  if [ -z "$RUN_ID" ]; then
    printf 'No run found (in this host'"'"'s view of outputs/) for experiment_id=%s\n' "$EXPERIMENT_ID"
    exit 0
  fi
else
  RUN_FILE="outputs/train_supernet/latest_run_id.$(hostname).txt"
  if [ ! -f "$RUN_FILE" ]; then
    printf 'No training run has been started from this host (%s). Pass an experiment_id to find a run started elsewhere: bash %s <experiment_id>\n' "$(hostname)" "$0"
    exit 0
  fi
  RUN_ID="$(cat "$RUN_FILE")"
fi

JOB_DIR="outputs/train_supernet/$RUN_ID"
printf 'RUN_ID=%s\n\nSTATE:\n' "$RUN_ID"
[ -f "$JOB_DIR/state.env" ] && cat "$JOB_DIR/state.env" || printf 'state file not created yet\n'
printf '\nPROCESS (this host, %s, only -- a process shown here reflects local work; state.env above reflects whatever host last wrote it):\n' "$(hostname)"
pgrep -af '[r]un_train_supernet.sh|[t]rain_supernet.py' || printf 'no matching process on %s\n' "$(hostname)"
printf '\nLOG TAIL:\n'
[ -f "$JOB_DIR/train.log" ] && tail -n 30 "$JOB_DIR/train.log" || printf 'no log yet\n'
