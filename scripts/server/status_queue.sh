#!/usr/bin/env bash
# Read-only status of this server's queue(s): processes (must be at most ONE training
# process), GPU, per-job state, last throughput/ETA line.
#   bash scripts/server/status_queue.sh            # all queues seen on NFS
set -uo pipefail
cd "${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
echo "== $(hostname) $(date -u +%FT%TZ)"
N=$(pgrep -fc "[t]rain_supernet.py|[t]rain_baseline.py" || true)
echo "training processes on this host: ${N:-0} (expected 0 or 1)"
pgrep -af "[t]rain_supernet.py|[t]rain_baseline.py" | cut -c1-160
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null
for Q in outputs/train_queue/*/; do
  [ -d "$Q" ] || continue
  echo "-- queue $(basename "$Q"): $(tail -n 1 "$Q/queue.log" 2>/dev/null)"
  for J in "$Q"*/; do
    [ -f "$J/state.env" ] || continue
    S=$(grep '^status=' "$J/state.env" | cut -d= -f2)
    L=$(grep -E 'step [0-9]+/' "$J/train.log" 2>/dev/null | tail -n 1 | sed 's/.*step /step /' | cut -c1-120)
    E=$(grep -iE 'Traceback|Error|nan' "$J/train.log" 2>/dev/null | tail -n 1 | cut -c1-120)
    echo "   $(basename "$J"): $S | ${L:-no step yet} ${E:+| ERR: $E}"
  done
done
