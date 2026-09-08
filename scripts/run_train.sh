#!/usr/bin/env bash
# Train IMUPoser on IMU4D shards. Override any variable from the environment, e.g.
#   DATASETS=motionmillion/v1,hiphi/v1 EXPERIMENT=imuposer_mm bash scripts/run_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS="${DATASETS:-humoto/v1}"
EXPERIMENT="${EXPERIMENT:-imuposer_$(echo "$DATASETS" | tr '/,' '__')}"
python scripts/train.py --experiment "$EXPERIMENT" --datasets "$DATASETS" \
    --max_epochs "${MAX_EPOCHS:-50}" --window_length "${WINDOW_LENGTH:-125}" \
    --batch_size "${BATCH_SIZE:-64}" --num_workers "${NUM_WORKERS:-4}" --logger "${LOGGER:-wandb}" "$@"
