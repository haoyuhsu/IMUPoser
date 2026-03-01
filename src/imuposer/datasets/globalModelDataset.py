import torch
from torch.utils.data import Dataset
from imuposer import math
from imuposer.config import Config, amass_combos
from tqdm import tqdm
import random

import sys
sys.path.append('/projects/illinois/eng/cs/shenlong/personals/haoyu/imu-humans/code/imu-human-mllm/imu_synthesis')
sys.path.append('/home/haoyuyh3/Documents/maxhsu/imu-humans/imu-human-mllm/imu_synthesis')
from get_imu_readings import simulate_imu_readings

class GlobalModelDataset(Dataset):
    def __init__(self, split="train", config:Config=None):
        super().__init__()

        # load the data
        self.train = split
        self.config = config
        self.combos = list(amass_combos.items())
        self.data = self._prepare_dataset()

    def _get_data_files(self):
        """Get data files based on split and dataset_name."""
        if self.train == "train":
            if self.config.dataset_name == "humanml":
                # return [f"humanml_train_{i:03d}.pt" for i in range(10)]
                return ["humanml_train.pt"]
            elif self.config.dataset_name == "lingo":
                # return ["lingo_train_000.pt"]
                return ["LINGO_train.pt"]
            elif self.config.dataset_name == "all_no_MotionGV":
                return [
                    "aist_train.pt",
                    "BABEL_train.pt",
                    "EgoBody_train.pt",
                    "finedance_train.pt",
                    "fit3d_train.pt",
                    "haa500_train.pt",
                    "hi4d_train.pt",
                    "humanml_train.pt",
                    "humansc3d_train.pt",
                    "idea400_train.pt",
                    "interhuman_train.pt",
                    "interx_train.pt",
                    "kungfu_train.pt",
                    "LINGO_train.pt",
                    "music_train.pt",
                    "PhantomDanceDatav1.1_train.pt",
                    "trumans_train.pt"
                ]
            elif self.config.dataset_name == "all":
                return [
                    "aist_train.pt",
                    "BABEL_train.pt",
                    "EgoBody_train.pt",
                    "finedance_train.pt",
                    "fit3d_train.pt",
                    "haa500_train.pt",
                    "hi4d_train.pt",
                    "humanml_train.pt",
                    "humansc3d_train.pt",
                    "idea400_train.pt",
                    "interhuman_train.pt",
                    "interx_train.pt",
                    "kungfu_train.pt",
                    "LINGO_train.pt",
                    "music_train.pt",
                    "PhantomDanceDatav1.1_train.pt",
                    "trumans_train.pt",
                    ### MotionGV ###
                    "Mirror_MotionGV_folder0_train.pt",
                    "Mirror_MotionGV_folder1_train.pt",
                    "Mirror_MotionGV_folder2_train.pt",
                    "Mirror_MotionGV_folder3_train.pt",
                    "Mirror_MotionGV_folder4_train.pt",
                    "Mirror_MotionGV_folder5_train.pt",
                    "Mirror_MotionGV_folder6_train.pt",
                    "Mirror_MotionGV_folder7_train.pt",
                    "Mirror_MotionGV_folder8_train.pt",
                    "Mirror_MotionGV_folder9_train.pt",
                ]
            else:
                raise ValueError(f"Unknown dataset_name: {self.config.dataset_name}")
        else:
            if self.config.dataset_name == "humanml":
                # return ["humanml_test_000.pt"]
                return ["humanml_test.pt"]
            elif self.config.dataset_name == "lingo":
                # return ["lingo_test_000.pt"]
                return ["lingo_test.pt"]
            elif self.config.dataset_name == "all" or self.config.dataset_name == "all_no_MotionGV":
                return [
                    # "aist_test.pt",
                    # "BABEL_test.pt",
                    "EgoBody_test.pt",   # save test set loading time for now
                    # "finedance_test.pt",
                    # "fit3d_test.pt",
                    # "haa500_test.pt",
                    # "hi4d_test.pt",
                    # "humanml_test.pt",
                    # "humansc3d_test.pt",
                    # "idea400_test.pt",
                    # "interhuman_test.pt",
                    # "interx_test.pt",
                    # "kungfu_test.pt",
                    # "LINGO_test.pt",
                    # "music_test.pt",
                    # "trumans_test.pt"
                ]
            else:
                raise ValueError(f"Unknown dataset_name: {self.config.dataset_name}")

    def _prepare_dataset(self):
        """Load raw data without combo-specific processing."""
        data_files = self._get_data_files()

        # Store raw data for runtime IMU simulation
        data = {
            'acc': [],          # acceleration for all 6 IMUs (if pre-computed, otherwise None)
            'ori': [],           # orientation for all 6 IMUs
            'pose': [],          # pose outputs
            'vpos': [],          # vertex positions for IMU simulation
            'fnames': []         # file names
        }

        # Minimum frames required for IMU simulation
        MIN_FRAMES = 4
        window_length = self.config.max_sample_len * 25 // 60 if self.train == 'train' else None

        for fname in tqdm(data_files, desc=f"Loading {self.train} data"):

            if self.train == 'train':
                fdata = torch.load(self.config.processed_imu_poser_25fps / fname)
            else:
                fdata = torch.load(self.config.processed_imu_poser_25fps / 'eval' / fname)

            n_samples = len(fdata["acc"])

            for i in range(n_samples):                
                # Get acceleration
                facc = fdata["acc"][i].view(-1, 6, 3)  # N, 6, 3
                                
                # Get orientation (always needed)
                fori = fdata["ori"][i].view(-1, 6, 3, 3)  # N, 6, 3, 3
                
                # Get vertex positions if available
                vpos = fdata.get('vpos', [None] * n_samples)[i]
                
                # Get pose
                fpose = fdata["pose"][i].reshape(fdata["pose"][i].shape[0], -1)
                
                # Get fname if available
                sample_fname = fdata.get('fname', [f"{fname}_{i}" for _ in range(n_samples)])[i]
                
                # Skip sequences that are too short
                if len(fori) < MIN_FRAMES:
                    continue
                
                if self.train == 'train' and window_length is not None:
                    # Use first K frames for training
                    data_len = min(window_length, len(fori))
                    data['acc'].append(facc[:data_len])
                    data['ori'].append(fori[:data_len])
                    data['pose'].append(fpose[:data_len])
                    data['vpos'].append(vpos[:data_len] if vpos is not None else None)
                    data['fnames'].append(sample_fname)
                else:
                    # Use entire sequence for evaluation
                    data['acc'].append(facc)
                    data['ori'].append(fori)
                    data['pose'].append(fpose)
                    data['vpos'].append(vpos)
                    data['fnames'].append(sample_fname)

            # Remove fdata to save memory
            del fdata
            import gc; gc.collect()
        
        return data
    

    def _apply_combo_mask(self, acc, ori, combo_indices):
        """Apply combo mask to acceleration and orientation data."""
        combo_acc = torch.zeros_like(acc)
        combo_ori = torch.zeros_like(ori)
        combo_acc[:, combo_indices] = acc[:, combo_indices]
        combo_ori[:, combo_indices] = ori[:, combo_indices]
        imu_input = torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)  # [N, 60]
        return imu_input


    # def load_data(self):
    #     # if self.train == "train":
    #     #     data_files = [x.name for x in self.config.processed_imu_poser_25fps.iterdir() if "dip" not in x.name]
    #     # else:
    #     #     data_files = ["dip_test.pt"]

    #     # TODO: change to HumanML dataset for custom training
    #     # if self.train == "train":
    #     #     data_files = ["humanml_train.pt"]
    #     # else:
    #     #     data_files = ["humanml_test.pt"]

    #     if self.train == "train":
    #         if self.config.dataset_name == "humanml":
    #             data_files = [f"humanml_train_{i:03d}.pt" for i in range(10)]
    #         elif self.config.dataset_name == "lingo":
    #             data_files = [f"lingo_train_000.pt"]
    #     else:
    #         if self.config.dataset_name == "humanml":
    #             data_files = ["humanml_test_000.pt"]
    #         elif self.config.dataset_name == "lingo":
    #             data_files = ["lingo_test_000.pt"]

    #     imu = []
    #     pose = []

    #     # window_length = self.config.max_sample_len * 25 // 60    # 300/60 = 5 seconds * 25 fps = 125 frames

    #     for fname in data_files:
    #         fdata = torch.load(self.config.processed_imu_poser_25fps / fname)

    #         n_samples = len(fdata["acc"])

    #         for i in tqdm(range(n_samples), dynamic_ncols=True):
    #             # inputs
    #             facc = fdata["acc"][i] 
    #             fori = fdata["ori"][i]

    #             # load all the data
    #             glb_acc = facc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]] / self.config.acc_scale
    #             glb_ori = fori.view(-1, 6, 3, 3)[:, [0, 1, 2, 3, 4]]

    #             # outputs
    #             fpose = fdata["pose"][i]
    #             fpose = fpose.reshape(fpose.shape[0], -1)

    #             window_length = self.config.max_sample_len * 25 // 60 if self.train == 'train' else len(glb_acc)

    #             # Split into windows WITHOUT applying combo masks
    #             acc_windows = torch.split(glb_acc, window_length)
    #             ori_windows = torch.split(glb_ori, window_length)
    #             pose_windows = torch.split(fpose, window_length)
                
    #             # Store each window once, we'll apply combos in __getitem__
    #             for acc_win, ori_win, pose_win in zip(acc_windows, ori_windows, pose_windows):
    #                 imu.append((acc_win, ori_win))
    #                 pose.append(pose_win)

    #         # remove fdata to save memory
    #         del fdata

    #     self.imu = imu
    #     self.pose = pose

    #     self.base_length = len(self.imu)


    def __getitem__(self, idx):

        # Get raw data
        ori = self.data['ori'][idx].float()  # N, 6, 3, 3
        _pose = self.data['pose'][idx].float()
        vpos = self.data['vpos'][idx]
        fname = self.data['fnames'][idx]

        if vpos is not None:
            # Runtime IMU simulation
            vpos_full = vpos.float()  # N, 6, 3
            ori_full = ori.float()    # N, 6, 3, 3
            
            # Simulate IMU readings for ALL 5 IMUs
            if self.train == 'train':
                # a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(   # use noisy simulation for training
                #     vpos_full, 
                #     ori_full, 
                #     fps=30,
                #     noise_raw_traj=True,
                #     noise_syn_imu=True,
                #     noise_est_orient=True,
                #     skip_ESKF=True,
                #     device='cpu'
                # )
                a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(   # use clean simulation for training
                    vpos_full, 
                    ori_full, 
                    fps=30,
                    noise_raw_traj=False,
                    noise_syn_imu=False,
                    noise_est_orient=False,
                    skip_ESKF=True,
                    device='cpu'
                )
            else:
                a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(   # use clean simulation for evaluation
                    vpos_full, 
                    ori_full, 
                    fps=30,
                    noise_raw_traj=False,
                    noise_syn_imu=False,
                    noise_est_orient=False,
                    skip_ESKF=True,
                    device='cpu'
                )
            
            # Normalize acceleration
            acc = a_sim[:, :5] / self.config.acc_scale  # N, 6, 3
            ori = R_sim[:, :5]  # N, 6, 3, 3
        else:
            # Use pre-computed accelerations and orientations
            # acc = self.data['acc'][idx][:, :5].float()
            # ori = self.data['ori'][idx][:, :5].float()
            raise ValueError(f"Vertex positions not available for {fname}, cannot simulate IMU readings.")
        
        # Select combo based on train/test
        if self.train == "train":
            combo_name, combo_indices = random.choice(self.combos)
        else:
            combo_indices = amass_combos["global"]
        
        # Apply combo mask
        _input = self._apply_combo_mask(acc, ori, combo_indices)
        
        # Prepare output
        if self.config.r6d:
            _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        else:
            _output = _pose

        return _input, _output


        # acc, ori = self.imu[idx]  # acc: N×5×3, ori: N×5×3×3
        # _pose = self.pose[idx].float()
        
        # # Randomly select a combo
        # if self.train == "train":
        #     # combo_name = random.choice(self.combos)
        #     # combo_mask = amass_combos[combo_name]
        #     combo_mask = random.choice(list(amass_combos.values()))
        # else:
        #     combo_mask = amass_combos["global"]
        
        # _combo_acc = torch.zeros_like(acc)
        # _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], 5, 1, 1)
        # _combo_acc[:, combo_mask] = acc[:, combo_mask]
        # _combo_ori[:, combo_mask] = ori[:, combo_mask]
        # _input = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1).float()
        
        # # Prepare output
        # if self.config.r6d == True:
        #     _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        # else:
        #     _output = _pose

        # return _input, _output


        # _imu = self.imu[idx].float()
        # _pose = self.pose[idx].float()

        # _input = _imu
        # if self.config.r6d == True:
        #     _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        # else:
        #     _output = _pose

        # return _input, _output

    def __len__(self):
        return len(self.data['ori'])

