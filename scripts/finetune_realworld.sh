#!/usr/bin/env bash
# Fine-tune a pretrained IMUPoser checkpoint separately on each real-IMU dataset (dipimu, imuposer, ncsa).
#   bash scripts/finetune_realworld.sh                                   # newest imuposer_pretrain run
#   PRETRAINED=checkpoints/<run>/<ckpt>.ckpt bash scripts/finetune_realworld.sh
#   REAL_DATASETS="imuposer/v2" MAX_EPOCHS=10 LOGGER=csv bash scripts/finetune_realworld.sh
# One experiment per dataset: checkpoints/imuposer_ft_<dataset>_global-<timestamp>/.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$REPO_ROOT"

PRETRAINED="${PRETRAINED:-$(latest_best_checkpoint "${PRETRAIN_EXPERIMENT:-imuposer_pretrain}_global")}"
[ -f "$PRETRAINED" ] || { echo "pretrained checkpoint not found: $PRETRAINED" >&2; exit 1; }
echo "fine-tuning from: $PRETRAINED"

for spec in ${REAL_DATASETS:-dipimu/v2 imuposer/v2 ncsa/v1}; do
    DATASET="$(filter_available_datasets "$spec")"
    TAG="$(echo "$DATASET" | tr '/' '_')"
    echo "=== fine-tuning on $DATASET"
    "$PYTHON" scripts/train.py \
        --experiment "imuposer_ft_$TAG" \
        --datasets "$DATASET" \
        --combo_id global \
        --pretrained_ckpt "$PRETRAINED" \
        --max_epochs "${MAX_EPOCHS:-30}" \
        --window_length "${WINDOW_LENGTH:-125}" \
        --batch_size "${BATCH_SIZE:-32}" \
        --num_workers "${NUM_WORKERS:-4}" \
        --lr "${LR:-5e-5}" \
        --logger "${LOGGER:-wandb}" \
        "$@"
done
