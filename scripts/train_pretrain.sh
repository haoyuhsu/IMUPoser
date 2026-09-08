#!/usr/bin/env bash
# Pretrain IMUPoser on the virtual-IMU IMU4D training sets (hiphi, humoto, motionmillion, omomo).
#   bash scripts/train_pretrain.sh                       # defaults below
#   MAX_EPOCHS=20 LOGGER=csv bash scripts/train_pretrain.sh --num_workers 8
# Datasets without local shards are skipped with a warning. Extra arguments go to scripts/train.py.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

PRETRAIN_DATASETS="${PRETRAIN_DATASETS:-hiphi/v1 humoto/v1 motionmillion/v1 omomo/v1}"
EXPERIMENT="${EXPERIMENT:-imuposer_pretrain}"
DATASETS="$(filter_available_datasets $PRETRAIN_DATASETS)"
echo "pretraining on: $DATASETS"

"$PYTHON" scripts/train.py \
    --experiment "$EXPERIMENT" \
    --datasets "$DATASETS" \
    --combo_id global \
    --max_epochs "${MAX_EPOCHS:-50}" \
    --window_length "${WINDOW_LENGTH:-125}" \
    --batch_size "${BATCH_SIZE:-64}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --lr "${LR:-3e-4}" \
    --logger "${LOGGER:-wandb}" \
    "$@"
