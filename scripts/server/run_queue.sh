#!/usr/bin/env bash
# Worker for start_queue.sh -- never invoke directly. Runs each queue line in order,
# one job at a time on this server's GPU. Skips a job whose final checkpoint exists.
set -uo pipefail
QUEUE="$1"; QDIR="$2"
WORKERS="${WORKERS:-12}"; AMP="${AMP:-true}"
COMMON=(--override "training.num_workers=$WORKERS" --override "training.amp=$AMP" --override "training.cudnn_benchmark=true")
STEPS=100000
if [ -n "${SMOKE:-}" ]; then STEPS=30; COMMON+=(--override training.checkpoint_interval_steps=30 --override training.log_interval_steps=10); fi
echo "queue start $(date -u +%FT%TZ) host=$(hostname) git=$(git rev-parse --short HEAD) workers=$WORKERS amp=$AMP steps=$STEPS"
grep -v '^\s*#' "$QUEUE" | grep -v '^\s*$' | while read -r KIND EXP SEED MODEL; do
  [ -n "${SMOKE:-}" ] && EXP="${EXP}_smoke"
  JOB="$QDIR/$EXP"; mkdir -p "$JOB"
  FINAL="outputs/$EXP/checkpoints/step_$(printf '%08d' "$STEPS").pt"
  if [ -f "$FINAL" ]; then echo "skip $EXP (already finished)"; printf 'status=DONE\n' >"$JOB/state.env"; continue; fi
  ARGS=(--config configs/experiment/default.yaml --override "experiment_id=$EXP" --override "seed=$SEED" --override "training.max_steps=$STEPS" "${COMMON[@]}")
  printf 'status=RUNNING\nkind=%s\nexp=%s\nseed=%s\nmodel=%s\nstarted_utc=%s\n' "$KIND" "$EXP" "$SEED" "${MODEL:-}" "$(date -u +%FT%TZ)" >"$JOB/state.env"
  echo "start $EXP $(date -u +%FT%TZ)"
  if [ "$KIND" = supernet ]; then
    python -u scripts/train_supernet.py "${ARGS[@]}" >"$JOB/train.log" 2>&1 </dev/null; RC=$?
  else
    python -u scripts/train_baseline.py --model "$MODEL" "${ARGS[@]}" >"$JOB/train.log" 2>&1 </dev/null; RC=$?
  fi
  ST=DONE; [ "$RC" -eq 0 ] || ST=FAILED
  printf 'status=%s\nkind=%s\nexp=%s\ntask_status=%s\nfinished_utc=%s\n' "$ST" "$KIND" "$EXP" "$RC" "$(date -u +%FT%TZ)" >"$JOB/state.env"
  echo "end $EXP status=$ST rc=$RC $(date -u +%FT%TZ)"
done
echo "queue end $(date -u +%FT%TZ)"
