#!/usr/bin/env bash
# Start a detached exported-subnet QAT fine-tuning run (scripts/train_exported_subnet.py
# -- the "exported-subnet" cells of the 2026-09-20 QAT 2x2 factorial screen,
# docs/COORDINATION_LOG.md open thread #1). Activate the right conda env first, then:
#   bash scripts/server/start_train_exported_subnet.sh <level> <fp32-checkpoint> [config.yaml] [extra train_exported_subnet.py args]
# e.g.
#   bash scripts/server/start_train_exported_subnet.sh large outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt configs/experiment/default.yaml \
#     --override experiment_id=qat_exported_large_dynamic_seed0 --qat
# Multiple --override values must each repeat the flag (argparse action="append").
# A leading "--" is optional and stripped if present.
# Check progress with status_train_exported_subnet.sh <level>. Job dir is keyed by
# level so several levels can run concurrently on the same or different servers
# without colliding -- never paste a full training loop directly into an interactive
# shell, this owns nohup/setsid so the SSH session can close without killing the job.
set -euo pipefail
REPO_DIR="${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
cd "$REPO_DIR"

LEVEL="${1:?elasticity level required, e.g. large}"
shift
FP32_CHECKPOINT="${1:?fp32 supernet checkpoint path required}"
shift

if [ -n "$(git status --porcelain)" ]; then
  printf 'Refusing to start: server worktree is not clean.\n'
  git status --short
  exit 1
fi
if pgrep -af "[t]rain_exported_subnet.py --level $LEVEL " >/dev/null; then
  printf 'A training run for level=%s is already active on this host.\n' "$LEVEL"
  exit 1
fi

CONFIG="${1:-configs/experiment/default.yaml}"
shift || true
# Strip a literal "--" separator if present -- see start_train_supernet.sh, which hit
# this same bug live (task_status=2, "unrecognized arguments: --").
if [ "${1:-}" = "--" ]; then
  shift
fi
RUN_ID="$(hostname)_exported_subnet_${LEVEL}_$(date -u +%Y%m%dT%H%M%SZ)"
JOB_DIR="outputs/train_exported_subnet/$LEVEL/$RUN_ID"
mkdir -p "$JOB_DIR"
# outputs/ is a *shared* NFS mount across SERVER-01..05 (docs/INFRA_OVERRIDE.md) --
# same level can run concurrently on different servers (e.g. the 2026-09-20 QAT 2x2
# screen's cells 3/4 both use --level large, on different hosts), so the "latest run"
# pointer must be per-(level, hostname), not per-level alone -- a shared per-level-only
# pointer file gets silently overwritten by whichever host launches later, exactly the
# same failure mode status_train_supernet.sh was fixed for on 2026-09-16. Pass an
# experiment_id to status_train_exported_subnet.sh to find a run started elsewhere.
printf '%s\n' "$RUN_ID" > "outputs/train_exported_subnet/$LEVEL/latest_run_id.$(hostname).txt"

nohup setsid bash scripts/server/run_train_exported_subnet.sh "$LEVEL" "$FP32_CHECKPOINT" "$RUN_ID" "$CONFIG" "$@" \
  >"$JOB_DIR/launcher.log" 2>&1 </dev/null &
printf 'Exported-subnet training started: level=%s run_id=%s pid=%s\n' "$LEVEL" "$RUN_ID" "$!"
