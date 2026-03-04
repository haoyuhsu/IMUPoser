r"""
Dataset for fine-tuning on real IMU datasets (DIP-IMU and/or IMUPoser).

Supports dataset_name:
  - "dip"           : DIP-IMU only (6 IMUs, selects first 5)
  - "imuposer_real" : IMUPoser real-world only (5 IMUs)
  - "dip_imuposer"  : Combined DIP-IMU + IMUPoser
"""

import torch
from torch.utils.data import Dataset
from imuposer import math
from imuposer.config import Config, amass_combos
from tqdm import tqdm
import random


class GlobalModelDatasetFineTuneRealIMU(Dataset):
    def __init__(self, split="train", config: Config = None):
        super().__init__()

        self.train = split
        self.config = config
        self.data = self._prepare_dataset()

    def _get_data_files(self):
        """Return list of (filename, n_imus) tuples based on split and dataset_name."""
        dataset_name = self.config.dataset_name

        if self.train == "train":
            if dataset_name == "dip":
                return [("dip_train.pt", 6)]
            elif dataset_name == "imuposer_real":
                return [("imuposer_train.pt", 5)]
            elif dataset_name == "dip_imuposer":
                return [("dip_train.pt", 6), ("imuposer_train.pt", 5)]
            else:
                raise ValueError(f"Unsupported dataset_name for FineTuneRealIMU: {dataset_name}")
        else:
            if dataset_name == "dip":
                return [("dip_test.pt", 6)]
            elif dataset_name == "imuposer_real":
                return [("imuposer_test.pt", 5)]
            elif dataset_name == "dip_imuposer":
                return [("dip_test.pt", 6), ("imuposer_test.pt", 5)]
            else:
                raise ValueError(f"Unsupported dataset_name for FineTuneRealIMU: {dataset_name}")

    def _prepare_dataset(self):
        """Load and window the data from real IMU datasets."""
        file_specs = self._get_data_files()
        data_root = self.config.processed_imu_poser_25fps / "eval"

        imu = []   # list of (acc, ori) tuples,  acc: W×5×3,  ori: W×5×3×3
        pose = []  # list of pose tensors,         pose: W×(24*9)

        window_length = self.config.max_sample_len * 25 // 60 if self.train == "train" else None

        for fname, n_imus in file_specs:
            fdata = torch.load(data_root / fname, map_location="cpu")
            n_samples = len(fdata["acc"])

            for i in tqdm(range(n_samples), desc=f"Loading {fname}", dynamic_ncols=True):
                # --- acceleration ---
                facc = fdata["acc"][i]                          # N×n_imus×3
                if n_imus == 6:
                    facc = facc.view(-1, 6, 3)[:, :5]          # keep first 5
                else:
                    facc = facc.view(-1, 5, 3)
                glb_acc = facc / self.config.acc_scale

                # --- orientation ---
                fori = fdata["ori"][i]                          # N×n_imus×3×3
                if n_imus == 6:
                    fori = fori.view(-1, 6, 3, 3)[:, :5]
                else:
                    fori = fori.view(-1, 5, 3, 3)

                # --- pose ---
                fpose = fdata["pose"][i].reshape(fdata["pose"][i].shape[0], -1)

                # window the sequence
                if self.train == "train" and window_length is not None:
                    acc_wins = torch.split(glb_acc, window_length)
                    ori_wins = torch.split(fori, window_length)
                    pose_wins = torch.split(fpose, window_length)
                    for aw, ow, pw in zip(acc_wins, ori_wins, pose_wins):
                        imu.append((aw, ow))
                        pose.append(pw)
                else:
                    imu.append((glb_acc, fori))
                    pose.append(fpose)

            del fdata

        self.imu = imu
        self.pose = pose

    def __getitem__(self, idx):
        acc, ori = self.imu[idx]       # acc: W×5×3,  ori: W×5×3×3
        _pose = self.pose[idx].float()

        # randomly select a combo during training; use all 5 IMUs at test time
        if self.train == "train":
            combo_mask = random.choice(list(amass_combos.values()))
        else:
            combo_mask = amass_combos["global"]

        # apply combo mask
        _combo_acc = torch.zeros_like(acc)
        _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], 5, 1, 1)
        _combo_acc[:, combo_mask] = acc[:, combo_mask]
        _combo_ori[:, combo_mask] = ori[:, combo_mask]
        _input = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1).float()

        # output
        if self.config.r6d:
            _output = (
                math.rotation_matrix_to_r6d(_pose)
                .reshape(-1, 24, 6)[:, self.config.pred_joints_set]
                .reshape(-1, 6 * len(self.config.pred_joints_set))
            )
        else:
            _output = _pose

        return _input, _output

    def __len__(self):
        return len(self.imu)
