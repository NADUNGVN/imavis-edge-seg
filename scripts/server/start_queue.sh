#!/usr/bin/env bash
# Start a detached sequential training queue on THIS server (landscape retraining,
# 2026-10-03). Activate the conda env first, then:
#   bash scripts/server/start_queue.sh configs/queues/landscape_server01.txt
#   SMOKE=1 bash scripts/server/start_queue.sh configs/queues/landscape_server01.txt   # 30-step smoke run
# Optional env: WORKERS (default 8), AMP (default true). Jobs never touch git; reports
# are committed once by hand after the whole batch (docs/RESOLUTION_ORIENTATION_FIX_20261003.md).
set -euo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"
QUEUE="${1:?queue file required}"
[ -f "$QUEUE" ] || { echo "no such queue file: $QUEUE"; exit 1; }
NAME="$(basename "$QUEUE" .txt)${SMOKE:+_smoke}"
if pgrep -af "[r]un_queue.sh $QUEUE" >/dev/null; then echo "queue $NAME already running on $(hostname)"; exit 1; fi
if pgrep -af "[t]rain_supernet.py|[t]rain_baseline.py" >/dev/null; then echo "another training process is active on $(hostname):"; pgrep -af "[t]rain_supernet.py|[t]rain_baseline.py"; exit 1; fi
QDIR="outputs/train_queue/$NAME"
mkdir -p "$QDIR"
nohup setsid env SMOKE="${SMOKE:-}" WORKERS="${WORKERS:-8}" AMP="${AMP:-true}" \
  bash scripts/server/run_queue.sh "$QUEUE" "$QDIR" >"$QDIR/queue.log" 2>&1 </dev/null &
echo "queue $NAME started on $(hostname) pid=$! log=$QDIR/queue.log"
