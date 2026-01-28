r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.

"""

# %load_ext autoreload
# %autoreload 2

import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
import glob

from imuposer.config import Config, amass_datasets
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math

config = Config(project_root_dir="../../")

def process_amass():
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        return acc

    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
    body_model = ParametricModel(config.og_smpl_model_path)

    try:
        processed = [fpath.name for fpath in (config.processed_imu_poser / "AMASS").iterdir()]
    except:
        processed = []

    for ds_name in amass_datasets:
        if ds_name in processed:
            continue
        data_pose, data_trans, data_beta, length = [], [], [], []
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(config.raw_amass_path, ds_name, '*/*_poses.npz')), dynamic_ncols=True):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

        if len(data_pose) == 0:
            print(f"AMASS dataset, {ds_name} not supported")
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)

        # include the left and right index fingers in the pose
        pose[:, 23] = pose[:, 37]     # right hand 
        pose = pose[:, :24].clone()   # only use body + right and left fingers

        # align AMASS global frame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(math.axis_angle_to_rotation_matrix(pose[:, 0])))

        print('Synthesizing IMU accelerations and orientations')
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
        for i, l in tqdm(list(enumerate(length)), dynamic_ncols=True):
            if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
            p = math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
            out_pose.append(pose[b:b + l].clone())  # N, 24, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
            out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
            out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
            b += l

        print('Saving')
        amass_dir = config.processed_imu_poser / "AMASS"
        amass_dir.mkdir(exist_ok=True, parents=True)
        ds_dir = amass_dir / ds_name
        ds_dir.mkdir(exist_ok=True)

        torch.save(out_pose, ds_dir / 'pose.pt')
        torch.save(out_shape, ds_dir / 'shape.pt')
        torch.save(out_tran, ds_dir / 'tran.pt')
        torch.save(out_joint, ds_dir / 'joint.pt')
        torch.save(out_vrot, ds_dir / 'vrot.pt')
        torch.save(out_vacc, ds_dir / 'vacc.pt')
        print('Synthetic AMASS dataset is saved at', str(ds_dir))


def process_dipimu(split="test"):
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        return acc
    
    imu_mask = [7, 8, 9, 10, 0, 2]
    if split == "test":
        test_split = ['s_09', 's_10']
    else:
        test_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    accs, oris, poses, trans, shapes, joints, vrots, vaccs = [], [], [], [], [], [], [], []
    
    body_model = ParametricModel(config.og_smpl_model_path)
    
    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(config.raw_dip_path, subject_name)):
            path = os.path.join(config.raw_dip_path, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            shape = torch.ones((10))
            tran = torch.zeros(pose.shape[0], 3) # dip-imu does not contain translations
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(tran.clone())  
                
                shapes.append(shape.clone()) # default shape
                
                # forward kinematics to get the joint position
                p = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
                vacc = _syn_acc(vert[:, vi_mask])
                vrot = grot[:, ji_mask]
                
                joints.append(joint)
                vaccs.append(vacc)
                vrots.append(vrot)
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))
                
    path_to_save = config.processed_imu_poser / f"DIP_IMU/{split}"
    path_to_save.mkdir(exist_ok=True, parents=True)
    
    torch.save(poses, path_to_save / 'pose.pt')
    torch.save(shapes, path_to_save / 'shape.pt')
    torch.save(trans, path_to_save / 'tran.pt')
    torch.save(joints, path_to_save / 'joint.pt')
    torch.save(vrots, path_to_save / 'vrot.pt')
    torch.save(vaccs, path_to_save / 'vacc.pt')
    torch.save(oris, path_to_save / 'oris.pt')
    torch.save(accs, path_to_save / 'accs.pt')
    
    print('Preprocessed DIP-IMU dataset is saved at', path_to_save)


def process_humanml(split='train'):
    def _syn_acc(v):
        r"""
        Synthesize accelerations from vertex positions.
        """
        # For 30 fps: acceleration formula using finite differences
        # acc[i] = (v[i-1] + v[i+1] - 2*v[i]) * fps^2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 900 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        return acc
    
    def smooth_avg(acc=None, s=3):
        nan_tensor = (torch.zeros((s // 2, acc.shape[1], acc.shape[2])) * torch.nan)
        acc = torch.cat((nan_tensor, acc, nan_tensor))
        tensors = []
        for i in range(s):
            L = acc.shape[0]
            tensors.append(acc[i:L-(s-i-1)])
        smoothed = torch.stack(tensors).nanmean(dim=0)
        return smoothed

    humanml_data_root = '/home/haoyuyh3/Downloads/humanml_smpl_files'
    data_dir = os.path.join(humanml_data_root, split)

    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])

    # left wrist, right wrist, left thigh, right thigh, head, pelvis
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
    body_model = ParametricModel(config.og_smpl_model_path)

    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    
    print(f'Processing HumanML {split} split')
    for pkl_file in tqdm(file_list, dynamic_ncols=True):
        file_path = os.path.join(data_dir, pkl_file)
        try:
            data = pickle.load(open(file_path, 'rb'))
        except:
            print(f'Failed to load {pkl_file}')
            continue
        
        smpl_params = data['smpl_params']
        
        # Extract SMPL parameters
        global_orient = torch.from_numpy(smpl_params['global_orient']).float()  # (N, 3)
        body_pose = torch.from_numpy(smpl_params['body_pose']).float()  # (N, 23, 3)
        transl = torch.from_numpy(smpl_params['transl']).float()  # (N, 3)
        # Use zero shape as requested
        shape = torch.zeros(10)  # (10,)
        
        # Combine global_orient and body_pose to get full pose (N, 24, 3)
        pose = torch.cat([global_orient.unsqueeze(1), body_pose], dim=1)  # (N, 24, 3)
        
        seq_len = pose.shape[0]
        
        # Skip sequences that are too short
        if seq_len <= 12:
            print(f'\tDiscard {pkl_file} with length {seq_len}')
            continue
        
        # Reshape pose for axis angle format
        pose_aa = pose.view(-1, 24, 3)  # (N, 24, 3)
        
        # Convert to rotation matrices for forward kinematics
        p = math.axis_angle_to_rotation_matrix(pose_aa).view(-1, 24, 3, 3)
        
        # Forward kinematics to get joints and vertices
        grot, joint, vert = body_model.forward_kinematics(p, shape, transl, calc_mesh=True)
        
        # Synthesize IMU accelerations from vertices
        vacc = _syn_acc(vert[:, vi_mask])  # N, 6, 3
        
        # Apply smoothing to accelerations (similar to 25fps preprocessing)
        vacc = smooth_avg(vacc, s=5)
        
        # Extract virtual IMU orientations
        vrot = grot[:, ji_mask]  # N, 6, 3, 3
        
        # Store outputs
        out_pose.append(pose_aa.clone())  # N, 24, 3
        out_tran.append(transl.clone())  # N, 3
        out_shape.append(shape.clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(vacc)  # N, 6, 3
        out_vrot.append(vrot)  # N, 6, 3, 3
    
    print(f'Saving {len(out_pose)} sequences')
    
    # Save in the IMUPoser format (as a single .pt file with dict)
    # Convert poses to rotation matrices
    fdata = {
        "joint": out_joint,
        "pose": [math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3) for p in out_pose],
        "shape": out_shape,
        "tran": out_tran,
        "acc": out_vacc,
        "ori": out_vrot
    }
    
    path_to_save_25fps = config.processed_imu_poser_25fps
    path_to_save_25fps.mkdir(exist_ok=True, parents=True)
    torch.save(fdata, path_to_save_25fps / f"humanml_{split}.pt")
    
    print(f'HumanML {split} dataset is saved at {path_to_save_25fps / f"humanml_{split}.pt"}')



if __name__ == '__main__':
    # process_dipimu(split="test")
    # process_dipimu(split="train")

    # process_amass()

    # process_humanml(split='train')
    process_humanml(split='test')