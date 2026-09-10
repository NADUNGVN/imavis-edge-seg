#!/usr/bin/env bash
# Worker for a detached baseline training run. Never invoke directly -- use
# start_train_baseline.sh, which owns nohup/setsid so the interactive shell never owns
# this job (docs/COLLABORATION_PROTOCOL.md "Detached long-running jobs"). Assumes the
# right conda env is already active in the parent shell that calls start_*.sh.
set -uo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
MODEL="${1:?model name required}"
RUN_ID="${2:?run id required}"
CONFIG="${3:-configs/experiment/default.yaml}"
shift 3 2>/dev/null || shift $#
cd "$REPO_DIR"

JOB_DIR="outputs/train_baseline/$MODEL/$RUN_ID"
STATE="$JOB_DIR/state.env"
LOG="$JOB_DIR/train.log"
SERVER_REPORT="reports/server/${RUN_ID}.md"
mkdir -p "$JOB_DIR" reports/server

write_state() {
  local status="$1" code="$2"
  printf 'run_id=%s\nmodel=%s\nstatus=%s\ntask_status=%s\nupdated_utc=%s\nlog=%s\n' \
    "$RUN_ID" "$MODEL" "$status" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

TASK_STATUS=0
write_state RUNNING 99
printf 'Baseline training started at %s (model=%s config=%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL" "$CONFIG" >"$LOG"
python scripts/train_baseline.py --model "$MODEL" --config "$CONFIG" "$@" >>"$LOG" 2>&1 || TASK_STATUS=$?

FINAL_STATUS=DONE
[ "$TASK_STATUS" -eq 0 ] || FINAL_STATUS=FAILED
write_state "$FINAL_STATUS" "$TASK_STATUS"

{
  printf '# Baseline training report\n\nrun_id=%s\nmodel=%s\nconfig=%s\ntask_status=%s\n\n## Log tail\n\n```text\n' \
    "$RUN_ID" "$MODEL" "$CONFIG" "$TASK_STATUS"
  tail -n 100 "$LOG" 2>/dev/null || true
  printf '\n```\n'
} >"$SERVER_REPORT"

git add "$SERVER_REPORT" && git commit -m "report: add $RUN_ID" && git pull --rebase origin main && git push origin main
printf 'run_id=%s task_status=%s\n' "$RUN_ID" "$TASK_STATUS" >>"$LOG"
exit "$TASK_STATUS"
