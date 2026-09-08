"""Turn one IMU4D shard sample into baseline tensors, entirely at runtime.

IMU4D stores ``motion_data_smpl85`` as
``[root axis-angle 0:3 | 21 SMPL-X body joints axis-angle 3:66 | zeros 66:72 | pelvis position 72:75 | betas 75:85]``
and either ``imu_traj (T,6,6)`` (joint rotation axis-angle + sensor position, virtual IMUs) or
measured ``imu_acc (T,6,3)`` / ``imu_ori (T,6,3,3)`` (real IMUs). Sensors are ordered
``left_hip, right_hip, left_ear, right_ear, left_elbow, right_elbow``; the baseline uses the
five slots ``lw, rw, lp, rp, h`` so the elbow-joint sensors fill the wrist slots, the hip
sensors the pocket slots, the left ear the head slot, and the right ear is dropped.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

from imuposer.imu4d.paths import ensure_imu_synthesis_importable

ensure_imu_synthesis_importable()
from imu_synthesis.get_imu_readings import simulate_imu_readings  # noqa: E402

FPS = 30
ACC_SCALE = 30.0
NUM_BODY_JOINTS = 22  # SMPL-X body: pelvis + 21 joints
NUM_IMU4D_SENSORS = 6
NUM_MODEL_IMUS = 5
MIN_FRAMES = 4  # the accelerometer finite difference needs four samples
IMU4D_SENSOR_NAMES = ("left_hip", "right_hip", "left_ear", "right_ear", "left_elbow", "right_elbow")
# IMU4D slot -> baseline slot [lw, rw, lp, rp, h, extra]
IMU4D_TO_BASELINE = [4, 5, 0, 1, 2, 3]
REAL_IMU_SOURCES = ("imuposer", "dipimu", "ncsa")


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula: axis-angle ``(..., 3)`` to rotation matrices ``(..., 3, 3)``."""
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(aa.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    sin, cos = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return eye + sin * K + (1.0 - cos) * (K @ K)


def smooth_avg(acc: np.ndarray, window: int = 3) -> np.ndarray:
    """Centred moving average over time with NaN-padded edges, as in IMU4D's real-IMU loader."""
    pad = np.full((window // 2,) + acc.shape[1:], np.nan, dtype=acc.dtype)
    padded = np.concatenate((pad, acc, pad), axis=0)
    stacked = np.stack([padded[i : len(padded) - (window - i - 1)] for i in range(window)], axis=0)
    return np.nanmean(stacked, axis=0).astype(acc.dtype)


def first_frame_alignment(root_rotmat0: torch.Tensor, transl0: torch.Tensor):
    """Return ``(inv_rotation (3,3), inv_translation (3,))`` that puts the first-frame pelvis at the
    origin and removes its yaw about +Y, exactly like IMU4D's ``align_poses_to_first_frame``."""
    yaw = torch.atan2(root_rotmat0[0, 2], root_rotmat0[2, 2])
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    yaw_matrix = torch.stack(
        [torch.stack([cos, torch.zeros_like(cos), sin]),
         torch.tensor([0.0, 1.0, 0.0], dtype=cos.dtype),
         torch.stack([-sin, torch.zeros_like(cos), cos])]
    )  # (3, 3)
    return yaw_matrix.transpose(0, 1), -transl0


def crop_bounds(num_frames: int, window_length: Optional[int], split: str, rng: random.Random):
    """Pick ``[start, end)``: a random window for training, the whole sequence otherwise."""
    if window_length is None or split != "train":
        return 0, num_frames
    start = rng.randint(0, num_frames - window_length)
    return start, start + window_length


def convert_sample(
    sample: dict,
    split: str,
    window_length: Optional[int],
    align_first_frame: bool,
    rng: random.Random,
) -> Optional[dict]:
    """Return baseline tensors for one shard sample, or ``None`` if it is too short for ``split``.

    Output keys: ``pose (T,22,3,3)`` local rotation matrices with the root at index 0,
    ``transl (T,3)`` pelvis position, ``acc (T,5,3)`` gravity-free world-frame acceleration divided by
    ``ACC_SCALE``, ``ori (T,5,3,3)`` sensor orientation in the world frame, ``fname`` and ``source``.
    """
    motion = np.asarray(sample["motion_data_smpl85"], dtype=np.float32)  # (T, 85)
    assert motion.ndim == 2 and motion.shape[1] == 85, f"bad smpl85 shape {motion.shape}"
    source = str(sample.get("__source__", sample.get("source", "")))
    is_real = source in REAL_IMU_SOURCES
    if is_real:
        num_frames = min(len(motion), len(sample["imu_acc"]), len(sample["imu_ori"]))
    else:
        num_frames = min(len(motion), len(sample["imu_traj"]))
    if num_frames < MIN_FRAMES or (split == "train" and window_length is not None and num_frames < window_length):
        return None

    start, end = crop_bounds(num_frames, window_length, split, rng)
    motion = motion[start:end]
    T = end - start

    aa = torch.from_numpy(motion[:, 0:66]).reshape(T, NUM_BODY_JOINTS, 3)
    pose = axis_angle_to_matrix(aa)  # (T, 22, 3, 3), index 0 = root orientation
    transl = torch.from_numpy(motion[:, 72:75])  # (T, 3) pelvis world position

    if is_real:
        acc6 = np.asarray(sample["imu_acc"], dtype=np.float32)[start:end]  # (T, 6, 3) gravity removed
        ori6 = torch.from_numpy(np.asarray(sample["imu_ori"], dtype=np.float32)[start:end])  # (T, 6, 3, 3)
        if not bool(sample.get("imu_acc_smoothed", False)):
            acc6 = smooth_avg(acc6, window=3)
        acc6 = torch.from_numpy(acc6)
        acc6, ori6 = acc6[:, IMU4D_TO_BASELINE], ori6[:, IMU4D_TO_BASELINE]
        imu_pos = None
    else:
        imu_traj = torch.from_numpy(np.asarray(sample["imu_traj"], dtype=np.float32)[start:end])  # (T, 6, 6)
        imu_rot = axis_angle_to_matrix(imu_traj[:, :, 0:3])[:, IMU4D_TO_BASELINE]  # (T, 6, 3, 3)
        imu_pos = imu_traj[:, :, 3:6][:, IMU4D_TO_BASELINE]  # (T, 6, 3)

    if align_first_frame:
        inv_rot, inv_transl = first_frame_alignment(pose[0, 0], transl[0])
        pose[:, 0] = inv_rot @ pose[:, 0]
        transl = (transl + inv_transl) @ inv_rot.transpose(0, 1)
        if is_real:
            acc6 = acc6 @ inv_rot.transpose(0, 1)
            ori6 = inv_rot @ ori6
        else:
            imu_pos = (imu_pos + inv_transl) @ inv_rot.transpose(0, 1)
            imu_rot = inv_rot @ imu_rot

    if not is_real:
        # Clean simulation: world-frame gravity-free acceleration and world-frame orientation.
        with torch.no_grad():
            acc6, _, ori6, _, _, _ = simulate_imu_readings(
                imu_pos, imu_rot, fps=FPS, noise_raw_traj=False, noise_syn_imu=False,
                noise_est_orient=False, skip_ESKF=True, device="cpu",
            )

    key = str(sample.get("id", sample.get("__key__", "sample")))
    fname = f"{sample.get('__dataset__', source).replace('/', '_')}_{key}".replace("/", "_")
    return {
        "pose": pose.float(),
        "transl": transl.float(),
        "acc": (acc6[:, :NUM_MODEL_IMUS] / ACC_SCALE).float(),  # (T, 5, 3)
        "ori": ori6[:, :NUM_MODEL_IMUS].float(),  # (T, 5, 3, 3)
        "fname": fname,
        "source": source,
    }


def apply_combo_mask(acc: torch.Tensor, ori: torch.Tensor, combo_indices) -> torch.Tensor:
    """Zero the IMU slots outside ``combo_indices`` and flatten to the ``(T, 60)`` model input."""
    combo_acc = torch.zeros_like(acc)
    combo_ori = torch.zeros_like(ori)
    combo_acc[:, combo_indices] = acc[:, combo_indices]
    combo_ori[:, combo_indices] = ori[:, combo_indices]
    return torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)
