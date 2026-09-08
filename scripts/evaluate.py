"""Run IMUPoser on an IMU4D split and export per-sample ``.npz`` files for ``metric.motion``.

Each file holds ``pred_rotations / gt_rotations (T,22,3,3)`` local rotations, ``pred_joints / gt_joints
(T,22,3)`` root-relative SMPL-X joint positions, ``gt_root_translation (T,3)`` and, because IMUPoser
does not predict translation, ``pred_root_translation`` filled with zeros.

Example:
    python scripts/evaluate.py --checkpoint checkpoints/<run>/<ckpt>.ckpt --datasets humoto/v1 --combo lw_rp_h
    python -m metric.motion --input-dir baseline/IMUPoser/predictions/humoto_v1/lw_rp_h --max-frames 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imuposer.config import Config, amass_combos, parse_dataset_specs  # noqa: E402
from imuposer.datasets.imu4dDataset import IMU4DDataset  # noqa: E402
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", default="humoto/v1", help="comma-separated IMU4D specs to evaluate")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--combo", default="global", choices=list(amass_combos.keys()))
    parser.add_argument("--output_dir", default=None, help="default: <repo>/predictions/<datasets>/<combo>")
    parser.add_argument("--no_align", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None, help="stop after this many samples (debug)")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Stream the split, run the model on full sequences, and write one ``.npz`` per sample."""
    args = parse_args()
    specs = parse_dataset_specs(args.datasets)
    config = Config(model="IMUPoser", project_root_dir=str(REPO_ROOT), mkdir=False, device=args.device,
                    datasets=specs, align_first_frame=not args.no_align)
    device = config.device

    model = IMUPoserModel.load_from_checkpoint(args.checkpoint, config=config, map_location=device)
    model.to(device).eval()
    if model.bodymodel is None:
        from imuposer.smpl.smplxModel import SMPLXBodyModel
        model.bodymodel = SMPLXBodyModel(config.smplx_model_path, device=device)
    model.bodymodel.device = device

    dataset = IMU4DDataset(args.split, config, combo=args.combo, return_meta=True)
    loader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers)

    tag = "+".join(s.replace("/", "_") for s in specs)
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "predictions" / tag / args.combo
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for imu, target, (fname, gt_pose, gt_transl) in tqdm(loader, total=dataset.num_samples, desc=f"{tag}/{args.combo}"):
        T = imu.shape[0]
        pred = model(imu.unsqueeze(0).to(device), [T])[0]  # (T, n_pose_output)
        pred_rot = model.pose_to_rotmat(pred)  # (T, 22, 3, 3)
        gt_rot = gt_pose.to(device)  # (T, 22, 3, 3)
        pred_joints = model.bodymodel.forward_kinematics(pred_rot)[1]  # (T, 22, 3)
        gt_joints = model.bodymodel.forward_kinematics(gt_rot)[1]
        np.savez(
            output_dir / f"{fname}.npz",
            pred_rotations=pred_rot.cpu().numpy().astype(np.float32),
            gt_rotations=gt_rot.cpu().numpy().astype(np.float32),
            pred_joints=pred_joints.cpu().numpy().astype(np.float32),
            gt_joints=gt_joints.cpu().numpy().astype(np.float32),
            pred_root_translation=np.zeros((T, 3), dtype=np.float32),
            gt_root_translation=gt_transl.numpy().astype(np.float32),
        )
        count += 1
        if args.max_samples is not None and count >= args.max_samples:
            break

    print(f"wrote {count} samples to {output_dir}")
    print(f"evaluate with: python -m metric.motion --input-dir {output_dir} --max-frames 60")


if __name__ == "__main__":
    main()
