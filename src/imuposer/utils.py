"""Argument parsing and small pose helpers."""

from __future__ import annotations

import argparse

import torch

from imuposer.config import NUM_BODY_JOINTS, Config, amass_combos


def get_parser() -> argparse.ArgumentParser:
    """CLI shared by ``scripts/train.py``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="experiment name (checkpoint dir + logger project)")
    parser.add_argument("--datasets", default="humoto/v1",
                        help="comma-separated IMU4D specs, e.g. 'motionmillion/v1,hiphi/v1' or 'imuposer/v2'")
    parser.add_argument("--combo_id", default="global", choices=list(amass_combos.keys()),
                        help="device slots fed to the network; training still samples random sub-combos")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--window_length", type=int, default=125, help="training clip length in frames (30 fps)")
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--num_workers", type=int, default=Config.num_workers)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--pretrained_ckpt", default=None, help="initialise from this checkpoint (fine-tuning)")
    parser.add_argument("--no_align", action="store_true", help="disable first-frame yaw/position alignment")
    parser.add_argument("--no_joint_loss", action="store_true", help="disable the SMPL-X joint-position loss")
    parser.add_argument("--device", default="0", help="CUDA index or 'cpu'")
    parser.add_argument("--logger", default="wandb", choices=["wandb", "csv"])
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser


def convert_subset_pose_to_full(config: Config, subset_pose_batch: torch.Tensor) -> torch.Tensor:
    """Pad ``(B, T, len(pred_joints_set), 3, 3)`` rotations to all 22 body joints with identity."""
    B, T = subset_pose_batch.shape[:2]
    subset = subset_pose_batch.reshape(B, T, len(config.pred_joints_set), 3, 3)
    full = torch.eye(3, device=subset.device, dtype=subset.dtype).repeat(B, T, NUM_BODY_JOINTS, 1, 1)
    full[:, :, config.pred_joints_set] = subset
    return full
