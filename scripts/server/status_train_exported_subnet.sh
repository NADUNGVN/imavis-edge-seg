#!/usr/bin/env bash
# Read-only status check for exported-subnet training run(s). No polling sleep -- run
# this once, read the answer, decide what's next.
#
#   bash scripts/server/status_train_exported_subnet.sh <level>                  # this host's own latest run at that level
#   bash scripts/server/status_train_exported_subnet.sh <level> <experiment_id>  # find a specific experiment_id's run, any host
#
# outputs/ is a *shared* NFS mount across SERVER-01..05 (docs/INFRA_OVERRIDE.md).
# With no experiment_id, this reports the CURRENT HOST's own latest launch at this
# level (a per-(level, hostname) pointer file), not whichever server anywhere launched
# most recently -- the 2026-09-20 QAT 2x2 screen ran cells 3 and 4 concurrently on two
# different hosts, both at --level large, and the first version of this script (keyed
# by level only) reported only whichever host launched last -- the same failure mode
# status_train_supernet.sh was fixed for on 2026-09-16. Pass the experiment_id
# explicitly to check a *specific* run regardless of which host it was launched from.
set -uo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"

LEVEL="${1:?elasticity level required, e.g. large}"
EXPERIMENT_ID="${2:-}"

if [ -n "$EXPERIMENT_ID" ]; then
  RUN_ID=""
  for log in $(ls -t "outputs/train_exported_subnet/$LEVEL"/*/train.log 2>/dev/null); do
    if grep -qE "experiment_id=${EXPERIMENT_ID}( |\$)" "$log" 2>/dev/null; then
      RUN_ID="$(basename "$(dirname "$log")")"
      break
    fi
  done
  if [ -z "$RUN_ID" ]; then
    printf 'No run found (in this host'"'"'s view of outputs/) for level=%s experiment_id=%s\n' "$LEVEL" "$EXPERIMENT_ID"
    exit 0
  fi
else
  RUN_FILE="outputs/train_exported_subnet/$LEVEL/latest_run_id.$(hostname).txt"
  if [ ! -f "$RUN_FILE" ]; then
    printf 'No training run at level=%s has been started from this host (%s). Pass an experiment_id to find a run started elsewhere: bash %s %s <experiment_id>\n' \
      "$LEVEL" "$(hostname)" "$0" "$LEVEL"
    exit 0
  fi
  RUN_ID="$(cat "$RUN_FILE")"
fi

JOB_DIR="outputs/train_exported_subnet/$LEVEL/$RUN_ID"
printf 'RUN_ID=%s\n\nSTATE:\n' "$RUN_ID"
[ -f "$JOB_DIR/state.env" ] && cat "$JOB_DIR/state.env" || printf 'state file not created yet\n'
printf '\nPROCESS (this host, %s, only -- a process shown here reflects local work; state.env above reflects whatever host last wrote it):\n' "$(hostname)"
pgrep -af '[r]un_train_exported_subnet.sh|[t]rain_exported_subnet.py' || printf 'no matching process on %s\n' "$(hostname)"
printf '\nLOG TAIL:\n'
[ -f "$JOB_DIR/train.log" ] && tail -n 30 "$JOB_DIR/train.log" || printf 'no log yet\n'
