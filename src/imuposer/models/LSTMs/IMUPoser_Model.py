"""IMUPoser network: 5 IMUs (acc + orientation) to SMPL-X body pose as 6D rotations."""

from __future__ import annotations

from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from imuposer.config import NUM_BODY_JOINTS, Config
from imuposer.math.angular import r6d_to_rotation_matrix
from imuposer.smpl.smplxModel import SMPLXBodyModel

from .RNN import RNN


def length_mask(lengths: List[int], max_len: int, device: torch.device) -> torch.Tensor:
    """Boolean ``(B, T)`` mask that is True on real (non-padded) frames."""
    idx = torch.arange(max_len, device=device)[None, :]  # (1, T)
    return idx < torch.as_tensor(lengths, device=device)[:, None]  # (B, T)


class IMUPoserModel(pl.LightningModule):
    """Bidirectional LSTM regressor with an optional SMPL-X joint-position loss.

    Input ``(B, T, 12 * num_imus)`` = per IMU 3 acc + 9 rotation-matrix entries; output
    ``(B, T, 6 * len(config.pred_joints_set))`` 6D local rotations (root included at index 0).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_input = 12 * len(config.joints_set)
        self.n_output_joints = len(config.pred_joints_set)
        self.rot_dim = 6 if config.r6d else 9
        self.n_pose_output = self.n_output_joints * self.rot_dim
        self.batch_size = config.batch_size
        self.lr = config.lr

        self.dip_model = RNN(n_input=self.n_input, n_output=self.n_pose_output, n_hidden=512, bidirectional=True)
        self.loss = nn.MSELoss() if config.loss_type == "mse" else nn.L1Loss()

        self.bodymodel = None
        if config.use_joint_loss:
            self.bodymodel = SMPLXBodyModel(config.smplx_model_path, device=config.device, skinning=False)

        self.save_hyperparameters()
        self.training_step_outputs: List[torch.Tensor] = []
        self.validation_step_outputs: List[torch.Tensor] = []

    def forward(self, imu_inputs: torch.Tensor, imu_lens: List[int]) -> torch.Tensor:
        """Return predicted pose ``(B, T, n_pose_output)``; padded frames are zeros."""
        pred_pose, _, _ = self.dip_model(imu_inputs, imu_lens)
        return pred_pose

    def pose_to_rotmat(self, pose: torch.Tensor) -> torch.Tensor:
        """``(N, n_pose_output)`` predictions to full local rotations ``(N, 22, 3, 3)`` (identity for unpredicted joints)."""
        N = pose.shape[0]
        if self.config.r6d:
            rot = r6d_to_rotation_matrix(pose.reshape(-1, 6)).view(N, self.n_output_joints, 3, 3)
        else:
            rot = pose.view(N, self.n_output_joints, 3, 3)
        if self.n_output_joints == NUM_BODY_JOINTS:
            return rot
        full = torch.eye(3, device=pose.device, dtype=pose.dtype).repeat(N, NUM_BODY_JOINTS, 1, 1)
        full[:, self.config.pred_joints_set] = rot
        return full

    def joint_positions(self, pose: torch.Tensor) -> torch.Tensor:
        """Root-relative SMPL-X joint positions ``(N, 22, 3)`` from ``(N, n_pose_output)`` poses."""
        return self.bodymodel.forward_kinematics(self.pose_to_rotmat(pose))[1]

    def shared_step(self, batch: Tuple) -> torch.Tensor:
        """Masked pose loss plus, if enabled, the masked joint-position loss."""
        imu_inputs, target_pose, input_lengths, _ = batch
        pred_pose = self(imu_inputs, input_lengths)  # (B, T, C)
        T = pred_pose.shape[1]
        target_pose = target_pose[:, :T, : self.n_pose_output]
        mask = length_mask(input_lengths, T, pred_pose.device)  # (B, T)

        pred_valid = pred_pose[mask]  # (N, C)
        target_valid = target_pose[mask]  # (N, C)
        loss = self.loss(pred_valid, target_valid)
        if self.config.use_joint_loss:
            loss = loss + self.loss(self.joint_positions(pred_valid), self.joint_positions(target_valid))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("training_step_loss", loss.item(), batch_size=self.batch_size)
        self.training_step_outputs.append(loss.detach())
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("validation_step_loss", loss.item(), batch_size=self.batch_size)
        self.validation_step_outputs.append(loss.detach())
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        self._log_epoch(self.training_step_outputs, "train_loss")
        self.training_step_outputs.clear()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch(self.validation_step_outputs, "val_loss")
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()

    def _log_epoch(self, losses: List[torch.Tensor], name: str) -> None:
        """Log the epoch mean of the collected step losses."""
        if losses:
            self.log(name, torch.stack(losses).mean(), prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
