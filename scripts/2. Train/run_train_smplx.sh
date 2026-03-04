#!/bin/bash

# Train IMUPoser with SMPL-X per-sequence data (on-the-fly loading)
# Uses the same model architecture (GlobalModelIMUPoser) but with SMPLXDataset

python "1. Train Global Model.py" \
    --combo_id 'global' \
    --experiment 'IMUPoserGlobalModel_smplx' \
    --max_epochs 50 \
    --dataset_name 'smplx'
