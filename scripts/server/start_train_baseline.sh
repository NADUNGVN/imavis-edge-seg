#!/usr/bin/env bash
# Start a detached required-baseline training run (RESEARCH_PLAN.md §7). Activate the
# right conda env first, then:
#   bash scripts/server/start_train_baseline.sh <model> [config.yaml] [-- extra train_baseline.py args]
# e.g. bash scripts/server/start_train_baseline.sh mobilenetv3_deeplabv3
# Check progress with status_train_baseline.sh <model>. Job dir is keyed by model name
# so several baselines can run concurrently on the same or different servers without
# colliding -- never paste a full training loop directly into an interactive shell,
# this owns nohup/setsid so the SSH session can close without killing the job.
set -euo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"

MODEL="${1:?model name required, e.g. mobilenetv3_deeplabv3}"
shift

if [ -n "$(git status --porcelain)" ]; then
  printf 'Refusing to start: server worktree is not clean.\n'
  git status --short
  exit 1
fi
if pgrep -af "[t]rain_baseline.py --model $MODEL " >/dev/null; then
  printf 'A training run for %s is already active.\n' "$MODEL"
  exit 1
fi

CONFIG="${1:-configs/experiment/default.yaml}"
shift || true
RUN_ID="$(hostname)_${MODEL}_$(date -u +%Y%m%dT%H%M%SZ)"
JOB_DIR="outputs/train_baseline/$MODEL/$RUN_ID"
mkdir -p "$JOB_DIR"
printf '%s\n' "$RUN_ID" > "outputs/train_baseline/$MODEL/latest_run_id.txt"

nohup setsid bash scripts/server/run_train_baseline.sh "$MODEL" "$RUN_ID" "$CONFIG" "$@" \
  >"$JOB_DIR/launcher.log" 2>&1 </dev/null &
printf 'Baseline training started: model=%s run_id=%s pid=%s\n' "$MODEL" "$RUN_ID" "$!"
