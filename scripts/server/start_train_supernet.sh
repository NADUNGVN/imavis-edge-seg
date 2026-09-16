#!/usr/bin/env bash
# Start a detached supernet training run. Activate the right conda env first, then:
#   bash scripts/server/start_train_supernet.sh [config.yaml] [extra train_supernet.py args]
# e.g. multiple --override values must each repeat the flag (argparse action="append"):
#   bash scripts/server/start_train_supernet.sh configs/experiment/default.yaml --override seed=1 --override experiment_id=pace_seg_v1_seed1
# A leading "--" is optional and stripped if present, but --override itself always needs
# to be repeated per key=value pair -- "--override a=1 b=2" passes "b=2" as an
# unrecognized positional, not a second override.
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
# Strip a literal "--" separator if present -- it's a hint for the human caller (see
# the usage comment above), not something train_supernet.py's argparse understands;
# forwarding it caused a real "unrecognized arguments: --" failure in practice
# (task_status=2, SERVER-01_train_20260911T001714Z).
if [ "${1:-}" = "--" ]; then
  shift
fi
RUN_ID="$(hostname)_train_$(date -u +%Y%m%dT%H%M%SZ)"
JOB_DIR="outputs/train_supernet/$RUN_ID"
mkdir -p "$JOB_DIR"
# outputs/ is a *shared* NFS mount across SERVER-01..05 (docs/INFRA_OVERRIDE.md) --
# write a per-hostname pointer (what status_train_supernet.sh reads by default) as
# well as the legacy shared one (kept for any single-server-at-a-time workflow),
# so a status check on THIS host reports THIS host's own latest launch, not
# whichever server anywhere launched most recently.
printf '%s\n' "$RUN_ID" > "outputs/train_supernet/latest_run_id.$(hostname).txt"
printf '%s\n' "$RUN_ID" > outputs/train_supernet/latest_run_id.txt

nohup setsid bash scripts/server/run_train_supernet.sh "$RUN_ID" "$CONFIG" "$@" \
  >"$JOB_DIR/launcher.log" 2>&1 </dev/null &
printf 'Training started: run_id=%s pid=%s\n' "$RUN_ID" "$!"
