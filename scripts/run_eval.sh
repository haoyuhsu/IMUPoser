#!/usr/bin/env bash
# Export predictions for the IMU4D motion metric with the 5-device and 3-device combos.
#   CHECKPOINT=checkpoints/<run>/<ckpt>.ckpt DATASETS=humoto/v1 bash scripts/run_eval.sh
#   DATASETS=imuposer/v2 bash scripts/run_eval.sh          # newest imuposer_ft_imuposer_v2 run
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"
DATASETS="${DATASETS:-humoto/v1}"
CHECKPOINT="${CHECKPOINT:-$(latest_best_checkpoint "imuposer_ft_$(echo "$DATASETS" | tr '/' '_')_global" 2>/dev/null || latest_best_checkpoint imuposer_pretrain_global)}"
echo "checkpoint: $CHECKPOINT"
for COMBO in ${COMBOS:-global lw_rp_h}; do
    "$PYTHON" scripts/evaluate.py --checkpoint "$CHECKPOINT" --datasets "$DATASETS" --combo "$COMBO" "$@"
done
