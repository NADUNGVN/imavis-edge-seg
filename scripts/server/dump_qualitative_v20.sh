#!/usr/bin/env bash
# Dump V20 qualitative-figure assets (input/GT/4 predictions/entropy per selected held-out
# image) from the landscape Run A checkpoint and push them back. Run on the server that
# holds outputs/pace_seg_landscape_supernet_seed0 and the datasets (conda env active):
#   bash scripts/server/dump_qualitative_v20.sh                      # first selection (19 images)
#   SET=stratified bash scripts/server/dump_qualitative_v20.sh       # route-class stratified selection
set -euo pipefail
cd "${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
git pull --rebase -q
if [ "${SET:-}" = stratified ]; then
  SEL=paper/figures/qualitative/selection_v20_stratified.json; OUT=reports/qualitative_assets_v20_stratified
else
  SEL=paper/figures/qualitative/selection_v20.json; OUT=reports/qualitative_assets_v20
fi
PYTHONPATH=src python -u scripts/dump_qualitative_assets_v20.py --selection "$SEL" --out "$OUT"
git add -f "$OUT"
git commit -q -m "report: V20 qualitative assets ${SET:-main} (landscape Run A, audited)"
git pull --rebase -q && git push -q
echo "DONE: pushed $OUT ($(ls "$OUT" | wc -l) files)"
