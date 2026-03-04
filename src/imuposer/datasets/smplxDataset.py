"""
On-the-fly dataset for SMPL-X per-sequence .pkl files.

Each .pkl file contains:
  - motion_data_smpl85: (N, 85) = transl(3) + pose(72, 24 joints * 3 axis-angle) + betas(10)
  - motion_data_smpl141: (N, 141) = SMPL-X native representation (unused here)
  - imu_traj: (N, 6, 6) = 6 IMUs x (rot_axis_angle(3) + position(3))
  - texts: optional text description

SMPL-X 6-IMU order: [left_hip(0), right_hip(1), left_ear(2), right_ear(3), left_elbow(4), right_elbow(5)]
IMUPoser  6-IMU order (internal): [lw(0), rw(1), lp(2), rp(3), h(4), extra(5)]

Semantic reorder from SMPL-X → IMUPoser:
  left_elbow(4) → slot 0 (left arm / lw)
  right_elbow(5) → slot 1 (right arm / rw)
  left_hip(0) → slot 2 (left pocket / lp)
  right_hip(1) → slot 3 (right pocket / rp)
  left_ear(2) → slot 4 (head / h)
  right_ear(3) → slot 5 (extra, passed to simulator but dropped for model input)
"""

import os
import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from imuposer import math
from imuposer.config import Config, amass_combos

import sys
sys.path.append('/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/code/imu-human-mllm/imu_synthesis')
from get_imu_readings import simulate_imu_readings

# Reorder indices: SMPL-X 6-IMU → IMUPoser 6-IMU convention
# [left_elbow, right_elbow, left_hip, right_hip, left_ear, right_ear]
SMPLX_TO_IMUPOSER_6 = [4, 5, 0, 1, 2, 3]

MIN_FRAMES = 4


class SMPLXDataset(Dataset):
    """
    Dataset that loads SMPL-X per-sequence .pkl files.
    Compatible with the existing IMUPoser training pipeline (same output format
    as GlobalModelDataset).

    Args:
        preload: If True, load all .pkl files into memory during __init__
                 (faster iteration, higher memory usage).
                 If False, load each .pkl on-the-fly in __getitem__
                 (slower iteration, lower memory usage).
    """

    def __init__(self, split="train", config: Config = None, preload=False):
        super().__init__()
        self.config = config
        self.split = split
        self.combos = list(amass_combos.items())
        self.preload = preload

        # Data directory
        data_dir = Path(config.smplx_data_path) / split
        if not data_dir.exists():
            raise FileNotFoundError(f"SMPL-X data directory not found: {data_dir}")

        # List all .pkl files (use os.listdir for speed on large directories)
        fnames = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

        # only use data from LINGO, and humanml
        valid_datasets = ["LINGO", "humanml"]
        fnames = [f for f in fnames if any(ds in f for ds in valid_datasets)]

        fnames.sort()
        self.data_dir = data_dir
        self.fnames = fnames

        # Window length for training (max_sample_len is in frames at 60fps)
        # SMPL-X data is at 30fps, so: max_sample_len * 30 / 60
        self.window_length = config.max_sample_len * 30 // 60 if split == "train" else None

        # Pre-load all data into memory if requested
        self._cache = None
        if preload:
            self._preload_all()

        print(f"SMPLXDataset [{split}]: {len(self.fnames)} sequences from {data_dir} (preload={preload})")

    def _preload_all(self):
        """Load all .pkl files into memory."""
        from tqdm import tqdm
        self._cache = []
        for fname in tqdm(self.fnames, desc=f"Pre-loading {self.split} data"):
            fpath = self.data_dir / fname
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            self._cache.append({
                'smpl85': torch.from_numpy(data['motion_data_smpl85']).float(),
                'imu_traj': torch.from_numpy(data['imu_traj']).float(),
            })

    def __len__(self):
        return len(self.fnames)

    def _load_sample(self, idx):
        """Return (smpl85, imu_traj) tensors for a given index."""
        if self._cache is not None:
            return self._cache[idx]['smpl85'], self._cache[idx]['imu_traj']
        # On-the-fly loading
        fpath = self.data_dir / self.fnames[idx]
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        return (torch.from_numpy(data['motion_data_smpl85']).float(),
                torch.from_numpy(data['imu_traj']).float())

    def __getitem__(self, idx):
        smpl85, imu_traj = self._load_sample(idx)
        N = smpl85.shape[0]

        # Handle too-short sequences: pad to MIN_FRAMES
        if N < MIN_FRAMES:
            pad_n = MIN_FRAMES - N
            smpl85 = torch.cat([smpl85, smpl85[-1:].expand(pad_n, -1)], dim=0)
            imu_traj = torch.cat([imu_traj, imu_traj[-1:].expand(pad_n, -1, -1)], dim=0)
            N = MIN_FRAMES

        # Truncate for training
        if self.split == "train" and self.window_length is not None:
            N = min(self.window_length, N)
            smpl85 = smpl85[:N]
            imu_traj = imu_traj[:N]

        # ---- IMU processing ----
        imu_rot_aa = imu_traj[:, :, :3]   # (N, 6, 3)
        imu_pos = imu_traj[:, :, 3:6]     # (N, 6, 3)

        # Convert axis-angle to rotation matrix
        imu_rot = math.axis_angle_to_rotation_matrix(
            imu_rot_aa.reshape(-1, 3)
        ).reshape(N, 6, 3, 3)

        # Reorder from SMPL-X convention to IMUPoser convention (all 6 IMUs)
        imu_pos = imu_pos[:, SMPLX_TO_IMUPOSER_6]   # (N, 6, 3)
        imu_rot = imu_rot[:, SMPLX_TO_IMUPOSER_6]   # (N, 6, 3, 3)

        # Simulate IMU readings (accelerometer + estimated orientation)
        # Use clean simulation (consistent with current GlobalModelDataset training)
        a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
            imu_pos, imu_rot,
            fps=30,
            noise_raw_traj=False,
            noise_syn_imu=False,
            noise_est_orient=False,
            skip_ESKF=True,
            device='cpu'
        )

        # Use first 5 IMUs (drop the 6th = right_ear)
        acc = a_sim[:, :5] / self.config.acc_scale   # (N, 5, 3)
        ori = R_sim[:, :5]                           # (N, 5, 3, 3)

        # Select combo (random for training, global for eval)
        if self.split == "train":
            _, combo_indices = random.choice(self.combos)
        else:
            combo_indices = amass_combos["global"]

        # Apply combo mask → model input
        _input = self._apply_combo_mask(acc, ori, combo_indices)

        # ---- Pose processing ----
        # smpl85 layout: transl(3) + pose(72 = 24 joints * 3 axis-angle) + betas(10)
        pose_aa = smpl85[:, 3:75].reshape(N, 24, 3)
        pose_rotmat = math.axis_angle_to_rotation_matrix(
            pose_aa.reshape(-1, 3)
        ).reshape(N, 24, 3, 3)

        if self.config.r6d:
            _output = math.rotation_matrix_to_r6d(pose_rotmat).reshape(-1, 24, 6)
            _output = _output[:, self.config.pred_joints_set].reshape(
                -1, 6 * len(self.config.pred_joints_set)
            )
        else:
            _output = pose_rotmat.reshape(N, -1)

        return _input, _output

    def _apply_combo_mask(self, acc, ori, combo_indices):
        """Apply combo mask to acceleration and orientation data."""
        combo_acc = torch.zeros_like(acc)
        combo_ori = torch.zeros_like(ori)
        combo_acc[:, combo_indices] = acc[:, combo_indices]
        combo_ori[:, combo_indices] = ori[:, combo_indices]
        return torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1).float()
