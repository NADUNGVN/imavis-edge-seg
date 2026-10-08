#!/usr/bin/env bash
# Dump the V20 qualitative-figure assets (19 held-out images x input/GT/4 predictions/entropy)
# from the landscape Run A checkpoint and push them back. Run on the server that holds
# outputs/pace_seg_landscape_supernet_seed0 and the datasets (conda env active):
#   bash scripts/server/dump_qualitative_v20.sh
# Takes ~1 min on GPU; output ~4 MB in reports/qualitative_assets_v20.
set -euo pipefail
cd "${IMAVIS_EDGE_SEG_REPO_DIR:-$HOME/Dung_TDTU/imavis-edge-seg}"
git pull --rebase -q
PYTHONPATH=src python -u scripts/dump_qualitative_assets_v20.py
git add -f reports/qualitative_assets_v20
git commit -q -m "report: V20 qualitative assets (landscape Run A, audited)"
git pull --rebase -q && git push -q
echo "DONE: pushed reports/qualitative_assets_v20 ($(ls reports/qualitative_assets_v20 | wc -l) files)"
