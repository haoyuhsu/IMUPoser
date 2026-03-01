"""
Evaluation script for IMUPoser model - saves predictions for motion metric evaluation.
"""

import torch
from pytorch_lightning import seed_everything
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
import argparse

from imuposer.config import Config, amass_combos
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
from imuposer.models.LSTMs.IMUPoser_Model_FineTune import IMUPoserModelFineTune
from imuposer.datasets.globalModelDataset import GlobalModelDataset
from imuposer.datasets.globalModelDatasetFineTuneDIP import GlobalModelDatasetFineTuneDIP
from imuposer.math.angular import r6d_to_rotation_matrix

seed_everything(42, workers=True)


# def create_full_length_dataset(dataset_class, config, split="test", combo_name="global"):
#     """
#     Create a dataset with full-length sequences (no chunking) and fixed IMU combo.
#     """
#     class FullLengthFixedComboDataset(dataset_class):
#         def __init__(self, split, config, combo_name):
#             super().__init__(split=split, config=config)
#             self.split = split
#             self.fixed_combo = amass_combos[combo_name]
            
#         def __getitem__(self, idx):
#             acc, ori = self.imu[idx]
#             _pose = self.pose[idx].float()
            
#             # Apply fixed combo mask
#             _combo_acc = torch.zeros_like(acc)
#             _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], 5, 1, 1)
#             _combo_acc[:, self.fixed_combo] = acc[:, self.fixed_combo]
#             _combo_ori[:, self.fixed_combo] = ori[:, self.fixed_combo]
#             _input = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1).float()
            
#             from imuposer import math
#             if self.config.r6d == True:
#                 _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
#             else:
#                 _output = _pose
            
#             return _input, _output
    
#     return FullLengthFixedComboDataset(split, config, combo_name)


def create_full_length_dataset(dataset_class, config, split="test", combo_name="global"):
    """
    Create a dataset with full-length sequences (no chunking) and fixed IMU combo.
    """
    class FullLengthFixedComboDataset(dataset_class):
        def __init__(self, split, config, combo_name):
            super().__init__(split=split, config=config)
            self.split = split
            self.fixed_combo = amass_combos[combo_name]
            
        def __getitem__(self, idx):
            # Get raw data from parent's data dict
            ori = self.data['ori'][idx].float()  # N, 6, 3, 3
            _pose = self.data['pose'][idx].float()
            vpos = self.data['vpos'][idx]
            
            if vpos is not None:
                # Runtime IMU simulation
                from imuposer.datasets.globalModelDataset import simulate_imu_readings
                
                vpos_full = vpos.float()  # N, 6, 3
                ori_full = ori.float()    # N, 6, 3, 3
                
                # Simulate IMU readings for ALL 5 IMUs
                a_sim, w_sim, R_sim, aS, wS, p_sim = simulate_imu_readings(
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
                acc = a_sim[:, :5] / self.config.acc_scale  # N, 5, 3
                ori = R_sim[:, :5]  # N, 5, 3, 3
            else:
                # Use pre-computed accelerations
                acc = self.data['acc'][idx][:, :5].float() / self.config.acc_scale
                ori = self.data['ori'][idx][:, :5].float()
            
            # Apply fixed combo mask
            _input = self._apply_combo_mask(acc, ori, self.fixed_combo)
            
            from imuposer import math
            if self.config.r6d == True:
                _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
            else:
                _output = _pose
            
            return _input, _output
    
    return FullLengthFixedComboDataset(split, config, combo_name)


@torch.no_grad()
def save_predictions(model, dataset, device, dataset_name="dataset", output_dir="../../predictions"):
    """
    Run inference and save predictions in motion metric format.
    Each sample saved as: {'orient': (N,3), 'transl': (N,3), 'pose': (N,21,3), 'betas': (N,10)}
    """
    model.eval()
    
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {dataset_name} ({len(dataset)} samples)...")
    
    for idx in tqdm(range(len(dataset)), desc=f"Inference on {dataset_name}"):
        imu_input, target_pose = dataset[idx]
        
        # Add batch dimension and move to device
        imu_input = imu_input.unsqueeze(0).to(device)
        
        # Get sequence length
        seq_len = imu_input.shape[1]
        
        # Forward pass
        pred_pose_r6d = model(imu_input, [seq_len])
        
        # Remove batch dimension and move to CPU
        pred_pose_r6d = pred_pose_r6d[0].cpu()  # (N, 144) for 24 joints in r6d
        target_pose_r6d = target_pose.cpu()      # (N, 144)
        
        # Convert r6d to rotation matrix
        pred_pose_rotmat = r6d_to_rotation_matrix(pred_pose_r6d.view(-1, 24, 6))  # (N, 24, 3, 3)
        target_pose_rotmat = r6d_to_rotation_matrix(target_pose_r6d.view(-1, 24, 6))  # (N, 24, 3, 3)
        
        # Convert rotation matrices to axis-angle
        from imuposer.math.angular import rotation_matrix_to_axis_angle
        pred_pose_aa = rotation_matrix_to_axis_angle(pred_pose_rotmat.view(-1, 3, 3)).view(seq_len, 24, 3)
        target_pose_aa = rotation_matrix_to_axis_angle(target_pose_rotmat.view(-1, 3, 3)).view(seq_len, 24, 3)
        
        # Split into global_orient (root) and body_pose (rest)
        pred_global_orient = pred_pose_aa[:, 0, :].numpy()  # (N, 3)
        pred_body_pose = pred_pose_aa[:, 1:, :].numpy()     # (N, 23, 3)
        
        target_global_orient = target_pose_aa[:, 0, :].numpy()  # (N, 3)
        target_body_pose = target_pose_aa[:, 1:, :].numpy()     # (N, 23, 3)
        
        # Create zero translation (IMUPoser doesn't predict translation)
        pred_transl = np.zeros((seq_len, 3), dtype=np.float32)
        target_transl = np.zeros((seq_len, 3), dtype=np.float32)
        
        # Create zero betas (shape parameters)
        pred_betas = np.zeros((seq_len, 10), dtype=np.float32)
        target_betas = np.zeros((seq_len, 10), dtype=np.float32)
        
        # Save prediction in the format expected by motion metric code
        pred_dict = {
            'orient': pred_global_orient.astype(np.float32),           # (N, 3)
            'transl': pred_transl,                                      # (N, 3)
            'pose': pred_body_pose.astype(np.float32),      # (N, 23, 3)
            'betas': pred_betas                                         # (N, 10)
        }
        
        gt_dict = {
            'orient': target_global_orient.astype(np.float32),         # (N, 3)
            'transl': target_transl,                                    # (N, 3)
            'pose': target_body_pose.astype(np.float32),    # (N, 23, 3)
            'betas': target_betas                                       # (N, 10)
        }
        
        # Save as pickle file
        save_file = output_path / f"sample_{idx:05d}.pkl"
        with open(save_file, 'wb') as f:
            pickle.dump({'pred': pred_dict, 'gt': gt_dict}, f)
    
    print(f"Saved {len(dataset)} samples to {output_path}")
    return output_path


def main():
    # Configuration
    device = torch.device("cuda:0")
    output_dir = "../../predictions"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['humanml', 'dip', 'lingo'], required=True)
    args = parser.parse_args()

    # Checkpoint paths
    # humanml_checkpoint = "../../checkpoints/IMUPoserGlobalModelThree_global-01292026-031253/epoch=epoch=20-val_loss=validation_step_loss=0.00983.ckpt"
    # lingo_checkpoint = "../../checkpoints/IMUPoserGlobalModel_lingo_global-01292026-161703/epoch=epoch=30-val_loss=validation_step_loss=0.00735.ckpt"
    # dip_checkpoint = "../../checkpoints/IMUPoserFineTuneDIPThree_global-01292026-032524/epoch=epoch=8-val_loss=validation_step_loss=0.01838.ckpt"

    # Use checkpoints trained on all datasets for final evaluation
    humanml_checkpoint = "../../checkpoints/IMUPoserGlobalModel_all_global-02102026-000922/epoch=epoch=15-val_loss=validation_step_loss=0.01269.ckpt"
    lingo_checkpoint = "../../checkpoints/IMUPoserGlobalModel_all_global-02102026-000922/epoch=epoch=15-val_loss=validation_step_loss=0.01269.ckpt"
    dip_checkpoint = ""

    if args.dataset == 'humanml':

        # Process HumanML test set with global IMU configuration
        print("\n" + "="*80)
        print("Processing HumanML Test Set (global IMU configuration)")
        print("="*80)
        
        # Create config for HumanML
        humanml_config = Config(
            model="GlobalModelIMUPoser",
            project_root_dir="../../",
            joints_set=amass_combos["global"],
            r6d=True,
            loss_type="mse",
            use_joint_loss=True,
            device="0",
            mkdir=False,
            dataset_name="humanml"
        )
        
        # Load HumanML model
        print(f"Loading HumanML model from {humanml_checkpoint}...")
        humanml_model = IMUPoserModel.load_from_checkpoint(humanml_checkpoint, config=humanml_config)
        humanml_model.to(device)
        humanml_model.eval()
        
        humanml_dataset = create_full_length_dataset(
            GlobalModelDataset, 
            humanml_config, 
            split="test", 
            combo_name="global"
        )
        
        humanml_save_path = save_predictions(
            humanml_model, humanml_dataset, device, 
            dataset_name="imuposer/humanml_global", 
            output_dir=output_dir
        )

    elif args.dataset == 'dip':
    
        # Process DIP-IMU test set with lw_rp_h IMU configuration
        print("\n" + "="*80)
        print("Processing DIP-IMU Test Set (lw_rp_h IMU configuration)")
        print("="*80)
        
        # Create config for DIP
        dip_config = Config(
            model="GlobalModelIMUPoser",
            project_root_dir="../../",
            joints_set=amass_combos['global'],
            r6d=True,
            loss_type="mse",
            use_joint_loss=True,
            device="0",
            mkdir=False
        )
        
        # Load pretrained model first
        print(f"Loading pretrained model...")
        dip_config.model = "GlobalModelIMUPoser"
        pretrained_model = IMUPoserModel.load_from_checkpoint(
            humanml_checkpoint,
            config=dip_config
        )
        
        # Now load fine-tuned checkpoint
        print(f"Loading fine-tuned DIP model from {dip_checkpoint}...")
        dip_config.model = "GlobalModelIMUPoserFineTuneDIP"
        dip_model = IMUPoserModelFineTune.load_from_checkpoint(
            dip_checkpoint, 
            config=dip_config,
            pretrained_model=pretrained_model
        )
        dip_model.to(device)
        dip_model.eval()


        #########################
        # DEBUGGING: Create config for HumanML
        # dip_config = Config(
        #     model="GlobalModelIMUPoser",
        #     project_root_dir="../../",
        #     joints_set=amass_combos["global"],
        #     r6d=True,
        #     loss_type="mse",
        #     use_joint_loss=True,
        #     device="0",
        #     mkdir=False
        # )
        
        # # Load HumanML model
        # print(f"Loading HumanML model from {humanml_checkpoint}...")
        # dip_model = IMUPoserModel.load_from_checkpoint(humanml_checkpoint, config=dip_config)
        # dip_model.to(device)
        # dip_model.eval()
        #########################

        
        dip_dataset = create_full_length_dataset(
            GlobalModelDatasetFineTuneDIP,
            dip_config,
            split="test",
            combo_name="lw_rp_h"   # real-world setting
        )
        
        dip_save_path = save_predictions(
            dip_model, dip_dataset, device,
            dataset_name="imuposer/dip_lw_rp_h",
            output_dir=output_dir
        )

    elif args.dataset == 'lingo':

        # Process LINGO test set with global IMU configuration
        print("\n" + "="*80)
        print("Processing LINGO Test Set (global IMU configuration)")
        print("="*80)
        
        # Create config for LINGO
        lingo_config = Config(
            model="GlobalModelIMUPoser",
            project_root_dir="../../",
            joints_set=amass_combos["global"],
            r6d=True,
            loss_type="mse",
            use_joint_loss=True,
            device="0",
            mkdir=False,
            dataset_name="lingo"
        )
        
        # Load LINGO model
        print(f"Loading LINGO model from {lingo_checkpoint}...")
        lingo_model = IMUPoserModel.load_from_checkpoint(lingo_checkpoint, config=lingo_config)
        lingo_model.to(device)
        lingo_model.eval()
        
        lingo_dataset = create_full_length_dataset(
            GlobalModelDataset, 
            lingo_config, 
            split="test", 
            combo_name="global"
        )
        
        lingo_save_path = save_predictions(
            lingo_model, lingo_dataset, device, 
            dataset_name="imuposer/lingo_global", 
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()