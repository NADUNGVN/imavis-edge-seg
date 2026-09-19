#!/usr/bin/env bash
# Read-only status check for exported-subnet training run(s). No polling sleep -- run
# this once, read the answer, decide what's next.
#   bash scripts/server/status_train_exported_subnet.sh [level]   # omit level to list all
set -uo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"

report_one() {
  local level="$1"
  local run_file="outputs/train_exported_subnet/$level/latest_run_id.txt"
  if [ ! -f "$run_file" ]; then
    printf '%s: no training run has been started.\n' "$level"
    return
  fi
  local run_id job_dir
  run_id="$(cat "$run_file")"
  job_dir="outputs/train_exported_subnet/$level/$run_id"
  printf '=== %s (RUN_ID=%s) ===\nSTATE:\n' "$level" "$run_id"
  [ -f "$job_dir/state.env" ] && cat "$job_dir/state.env" || printf 'state file not created yet\n'
  printf '\nLOG TAIL:\n'
  [ -f "$job_dir/train.log" ] && tail -n 15 "$job_dir/train.log" || printf 'no log yet\n'
  printf '\n'
}

printf 'PROCESSES:\n'
pgrep -af '[r]un_train_exported_subnet.sh|[t]rain_exported_subnet.py' || printf 'no matching process\n'
printf '\n'

if [ -n "${1:-}" ]; then
  report_one "$1"
else
  if [ ! -d outputs/train_exported_subnet ]; then
    printf 'No exported-subnet training run has been started.\n'
    exit 0
  fi
  for level_dir in outputs/train_exported_subnet/*/; do
    [ -d "$level_dir" ] || continue
    report_one "$(basename "$level_dir")"
  done
fi
