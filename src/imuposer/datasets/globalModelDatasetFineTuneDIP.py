import torch
from torch.utils.data import Dataset
from imuposer import math
from imuposer.config import Config, amass_combos
from tqdm import tqdm
import random

class GlobalModelDatasetFineTuneDIP(Dataset):
    def __init__(self, split="train", config:Config=None):
        super().__init__()

        # load the data
        self.train = split
        self.config = config
        self.data = self.load_data()
        
    def load_data(self):
        if self.train == "train":
            data_files = ["dip_train.pt"]
        else:
            data_files = ["dip_test.pt"]

        imu = []
        pose = []

        for fname in data_files:
            fdata = torch.load(self.config.processed_imu_poser_25fps / fname)

            for i in tqdm(range(len(fdata["acc"])), dynamic_ncols=True):
                # inputs
                facc = fdata["acc"][i] 
                fori = fdata["ori"][i]

                # load all the data
                glb_acc = facc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]] / self.config.acc_scale
                glb_ori = fori.view(-1, 6, 3, 3)[:, [0, 1, 2, 3, 4]]

                # outputs
                fpose = fdata["pose"][i]
                fpose = fpose.reshape(fpose.shape[0], -1)

                window_length = self.config.max_sample_len * 25 // 60 if self.train == 'train' else len(glb_acc)

                # Split into windows WITHOUT applying combo masks
                acc_windows = torch.split(glb_acc, window_length)
                ori_windows = torch.split(glb_ori, window_length)
                pose_windows = torch.split(fpose, window_length)
                
                # Store each window once, we'll apply combos in __getitem__
                for acc_win, ori_win, pose_win in zip(acc_windows, ori_windows, pose_windows):
                    # Each window will be repeated len(self.combos) times in __getitem__
                    imu.append((acc_win, ori_win))
                    pose.append(pose_win)

            del fdata

        self.imu = imu
        self.pose = pose

    def __getitem__(self, idx):

        acc, ori = self.imu[idx]  # acc: N×5×3, ori: N×5×3×3
        _pose = self.pose[idx].float()
        
        # Randomly select a combo
        if self.train == "train":
            # combo_name = random.choice(self.combos)
            # combo_mask = amass_combos[combo_name]
            combo_mask = random.choice(list(amass_combos.values()))
        else:
            combo_mask = amass_combos["global"]
        
        _combo_acc = torch.zeros_like(acc)
        _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], 5, 1, 1)
        _combo_acc[:, combo_mask] = acc[:, combo_mask]
        _combo_ori[:, combo_mask] = ori[:, combo_mask]
        _input = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1).float()

        if self.config.r6d == True:
            _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        else:
            _output = _pose

        return _input, _output

    def __len__(self):
        return len(self.imu)

