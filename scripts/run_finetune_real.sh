#!/usr/bin/env bash
# Fine-tune a pretrained checkpoint on a real-IMU IMU4D dataset (imuposer/v2, dipimu/v2, ncsa/v1).
#   PRETRAINED=checkpoints/<run>/<ckpt>.ckpt DATASETS=imuposer/v2 bash scripts/run_finetune_real.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PRETRAINED:?set PRETRAINED to the checkpoint to start from}"
DATASETS="${DATASETS:-imuposer/v2}"
python scripts/train.py --experiment "imuposer_ft_$(echo "$DATASETS" | tr '/,' '__')" --datasets "$DATASETS" \
    --pretrained_ckpt "$PRETRAINED" --lr "${LR:-5e-5}" --max_epochs "${MAX_EPOCHS:-30}" \
    --batch_size "${BATCH_SIZE:-32}" --logger "${LOGGER:-wandb}" "$@"
