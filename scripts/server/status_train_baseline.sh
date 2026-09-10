#!/usr/bin/env bash
# Read-only status check for baseline training run(s). No polling sleep -- run this
# once, read the answer, decide what's next.
#   bash scripts/server/status_train_baseline.sh [model]   # omit model to list all
set -uo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"

report_one() {
  local model="$1"
  local run_file="outputs/train_baseline/$model/latest_run_id.txt"
  if [ ! -f "$run_file" ]; then
    printf '%s: no training run has been started.\n' "$model"
    return
  fi
  local run_id job_dir
  run_id="$(cat "$run_file")"
  job_dir="outputs/train_baseline/$model/$run_id"
  printf '=== %s (RUN_ID=%s) ===\nSTATE:\n' "$model" "$run_id"
  [ -f "$job_dir/state.env" ] && cat "$job_dir/state.env" || printf 'state file not created yet\n'
  printf '\nLOG TAIL:\n'
  [ -f "$job_dir/train.log" ] && tail -n 15 "$job_dir/train.log" || printf 'no log yet\n'
  printf '\n'
}

printf 'PROCESSES:\n'
pgrep -af '[r]un_train_baseline.sh|[t]rain_baseline.py' || printf 'no matching process\n'
printf '\n'

if [ -n "${1:-}" ]; then
  report_one "$1"
else
  if [ ! -d outputs/train_baseline ]; then
    printf 'No baseline training run has been started.\n'
    exit 0
  fi
  for model_dir in outputs/train_baseline/*/; do
    [ -d "$model_dir" ] || continue
    report_one "$(basename "$model_dir")"
  done
fi
