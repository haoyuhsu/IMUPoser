"""IMUPoser dataset streamed from IMU4D WDS shards with all features computed at runtime."""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

from imuposer.config import Config, amass_combos
from imuposer.imu4d.sample import apply_combo_mask, convert_sample
from imuposer.imu4d.wds_reader import IMU4DShardDataset
from imuposer.math.angular import rotation_matrix_to_r6d


class IMU4DDataset(IMU4DShardDataset):
    """Yield ``(imu (T,60), pose_r6d (T, 6*len(pred_joints_set)))`` per clip.

    Training clips are random ``config.window_length`` windows with a random device combo; other
    splits use the full sequence with ``combo`` (default ``global``). With ``return_meta=True`` each
    item also carries ``(fname, pose_rotmat (T,22,3,3), transl (T,3))`` for evaluation.
    """

    def __init__(self, split: str, config: Config, combo: Optional[str] = None, return_meta: bool = False, seed: int = 0):
        self.config = config
        self.split = split
        self.return_meta = return_meta
        self.combos = list(amass_combos.items())
        self.eval_combo = amass_combos[combo if combo is not None else "global"]
        self.rng = random.Random(seed)
        super().__init__(
            dataset_specs=config.datasets,
            split=split,
            transform=self._transform,
            shuffle=(split == "train"),
            shuffle_buffer=config.shuffle_buffer,
            seed=seed,
        )

    def _transform(self, sample: dict):
        """Convert one raw shard sample into the model input / target tensors."""
        window = self.config.window_length if self.split == "train" else None
        data = convert_sample(sample, self.split, window, self.config.align_first_frame, self.rng)
        if data is None:
            return None

        if self.split == "train":
            _, combo_indices = self.rng.choice(self.combos)
        else:
            combo_indices = self.eval_combo
        imu = apply_combo_mask(data["acc"], data["ori"], combo_indices)  # (T, 60)

        pose = data["pose"]  # (T, 22, 3, 3) local rotations incl. root
        if self.config.r6d:
            target = rotation_matrix_to_r6d(pose).reshape(pose.shape[0], -1, 6)[:, self.config.pred_joints_set]
            target = target.reshape(pose.shape[0], -1)  # (T, 6 * num_pred_joints)
        else:
            target = pose[:, self.config.pred_joints_set].reshape(pose.shape[0], -1)  # (T, 9 * num_pred_joints)

        if self.return_meta:
            return imu, target, (data["fname"], pose, data["transl"])
        return imu, target
