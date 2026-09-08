#!/usr/bin/env bash
# Export predictions for the IMU4D motion metric with the 5-device and 3-device combos.
#   CHECKPOINT=checkpoints/<run>/<ckpt>.ckpt DATASETS=humoto/v1 bash scripts/run_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CHECKPOINT:?set CHECKPOINT}"
DATASETS="${DATASETS:-humoto/v1}"
for COMBO in ${COMBOS:-global lw_rp_h}; do
    python scripts/evaluate.py --checkpoint "$CHECKPOINT" --datasets "$DATASETS" --combo "$COMBO" "$@"
done
