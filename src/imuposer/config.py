"""IMUPoser configuration for training on IMU4D SMPL-X shards.

Datasets are IMU4D ``'<dataset>/<version>'`` specs streamed straight from the WDS shards
(see ``imuposer.imu4d``); nothing is preprocessed to disk. The body model is SMPL-X neutral and the
network predicts the 22 SMPL-X body joints.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from imuposer.imu4d.paths import smplx_model_path

NUM_BODY_JOINTS = 22
FPS = 30

imuName2idx = {"lw": 0, "rw": 1, "lp": 2, "rp": 3, "h": 4}

# Every subset of the five device slots used for training-time augmentation and evaluation.
amass_combos = {
    "global": [0, 1, 2, 3, 4],
    "lw_rw_h": [0, 1, 4],
    "rw_lp_rp": [1, 2, 3],
    "lw_rw_rp": [0, 1, 3],
    "lw_rp_h": [0, 3, 4],
    "rw_rp_h": [1, 3, 4],
    "lw_lp_rp": [0, 2, 3],
    "lw_rw_lp": [0, 1, 2],
    "lw_lp_h": [0, 2, 4],
    "rw_lp_h": [1, 2, 4],
    "lw_rw": [0, 1],
    "lw_lp": [0, 2],
    "lw_rp": [0, 3],
    "lw_h": [0, 4],
    "rw_lp": [1, 2],
    "rw_rp": [1, 3],
    "rw_h": [1, 4],
    "lp_rp": [2, 3],
    "lp_h": [2, 4],
    "rp_h": [3, 4],
    "lw": [0],
    "rw": [1],
    "lp": [2],
    "rp": [3],
    "h": [4],
}

# SMPL-X body joint groups (same indices as SMPL joints 0-21).
pred_joints_set = {
    "legs": [0, 1, 2, 4, 5, 7, 8, 10, 11],
    "upper_body": [0, 3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21],
    "head": [0, 12, 15],
}

leaf_joints = [20, 21, 7, 8, 12]

# Real-IMU IMU4D sources; everything else carries virtual-IMU trajectories.
REAL_IMU_DATASETS = {"imuposer/v2", "dipimu/v2", "ncsa/v1", "ncsa/v1_heading_fix_by_visual_oracle"}


class Config:
    """All run-time settings; hyperparameters live here, not in the scripts."""

    max_sample_len = 300  # legacy 60 fps window; window_length below is what is actually used
    acc_scale = 30
    batch_size = 64
    torch_seed = 0
    num_workers = 4
    shuffle_buffer = 1000
    lr = 3e-4

    def __init__(
        self,
        experiment: Optional[str] = None,
        model: Optional[str] = None,
        project_root_dir: Optional[str] = None,
        joints_set: Optional[Sequence[int]] = None,
        loss_type: str = "mse",
        mkdir: bool = True,
        normalize: str = "no_translation",
        r6d: bool = True,
        device: Optional[str] = None,
        use_joint_loss: bool = True,
        pred_joints_set: Optional[Sequence[int]] = None,
        datasets: Optional[Sequence[str]] = None,
        window_length: int = 125,
        align_first_frame: bool = True,
        smplx_model: Optional[str] = None,
    ):
        self.experiment = experiment
        self.model = model
        self.root_dir = Path(project_root_dir if project_root_dir is not None else ".").absolute()
        self.joints_set = list(joints_set) if joints_set is not None else amass_combos["global"]
        self.pred_joints_set = list(pred_joints_set) if pred_joints_set is not None else list(range(NUM_BODY_JOINTS))
        self.loss_type = loss_type
        self.mkdir = mkdir
        self.normalize = normalize
        self.r6d = r6d
        self.use_joint_loss = use_joint_loss

        # IMU4D data
        self.datasets: List[str] = list(datasets) if datasets else ["humoto/v1"]
        self.window_length = int(window_length)  # frames at 30 fps used for each training clip
        self.align_first_frame = align_first_frame
        self.fps = FPS

        self.smplx_model_path = Path(smplx_model) if smplx_model else smplx_model_path()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif "cpu" in device:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        self.build_paths()

    def build_paths(self) -> None:
        """Create the checkpoint directory ``<root>/checkpoints/<experiment>-<timestamp>``."""
        if self.mkdir and self.experiment is not None:
            datestring = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
            self.checkpoint_path = self.root_dir / f"checkpoints/{self.experiment}-{datestring}"
            self.checkpoint_path.mkdir(exist_ok=True, parents=True)

    def summary(self) -> str:
        """Human-readable dump of the fields that matter for a run."""
        fields = ["experiment", "model", "datasets", "window_length", "align_first_frame", "joints_set",
                  "pred_joints_set", "loss_type", "use_joint_loss", "r6d", "batch_size", "lr", "device",
                  "smplx_model_path"]
        return "\n".join(f"  {k}: {getattr(self, k)}" for k in fields)


def parse_dataset_specs(spec: str) -> List[str]:
    """Split ``'humoto/v1,hiphi/v1'`` into a list of IMU4D dataset specs."""
    specs = [s.strip() for s in spec.split(",") if s.strip()]
    assert specs, f"no dataset given in {spec!r}"
    return specs
