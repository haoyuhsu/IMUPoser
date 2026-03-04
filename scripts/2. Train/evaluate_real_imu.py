"""
Evaluate a fine-tuned IMUPoser model on DIP-IMU and/or IMUPoser real-world test sets.

Usage examples:

  # Evaluate fine-tuned model on DIP-IMU test set
  python evaluate_real_imu.py \
      --dataset dip \
      --pretrained_ckpt /path/to/pretrained_base.ckpt \
      --finetuned_ckpt /path/to/finetuned.ckpt

  # Evaluate fine-tuned model on IMUPoser test set
  python evaluate_real_imu.py \
      --dataset imuposer_real \
      --pretrained_ckpt /path/to/pretrained_base.ckpt \
      --finetuned_ckpt /path/to/finetuned.ckpt

  # Evaluate fine-tuned model on both test sets
  python evaluate_real_imu.py \
      --dataset dip_imuposer \
      --pretrained_ckpt /path/to/pretrained_base.ckpt \
      --finetuned_ckpt /path/to/finetuned.ckpt

  # Evaluate pre-trained model directly (no fine-tuning) on DIP test set
  python evaluate_real_imu.py \
      --dataset dip \
      --pretrained_ckpt /path/to/pretrained_base.ckpt \
      --no_finetune
"""

import torch
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
from imuposer.models.LSTMs.IMUPoser_Model_FineTune import IMUPoserModelFineTune
from imuposer.datasets.globalModelDatasetFineTuneRealIMU import GlobalModelDatasetFineTuneRealIMU
from imuposer.math.angular import r6d_to_rotation_matrix, rotation_matrix_to_axis_angle
from imuposer import math as imuposer_math

seed_everything(42, workers=True)


# ─── dataset wrapper: full-length sequences with a fixed IMU combo ──────────
class FullLengthRealIMUDataset(GlobalModelDatasetFineTuneRealIMU):
    """Wraps GlobalModelDatasetFineTuneRealIMU so that __getitem__ returns
    full-length sequences with a fixed combo and also the sample index."""

    def __init__(self, config, combo_name="global"):
        super().__init__(split="test", config=config)
        self.fixed_combo = amass_combos[combo_name]

    def __getitem__(self, idx):
        acc, ori = self.imu[idx]
        _pose = self.pose[idx].float()

        # fixed combo mask
        _combo_acc = torch.zeros_like(acc)
        _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], 5, 1, 1)
        _combo_acc[:, self.fixed_combo] = acc[:, self.fixed_combo]
        _combo_ori[:, self.fixed_combo] = ori[:, self.fixed_combo]
        _input = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1).float()

        if self.config.r6d:
            _output = (
                imuposer_math.rotation_matrix_to_r6d(_pose)
                .reshape(-1, 24, 6)[:, self.config.pred_joints_set]
                .reshape(-1, 6 * len(self.config.pred_joints_set))
            )
        else:
            _output = _pose

        return _input, _output, f"sample_{idx:05d}"


# ─── inference + save ───────────────────────────────────────────────────────
@torch.no_grad()
def save_predictions(model, dataset, device, dataset_name, output_dir):
    model.eval()
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference on {dataset_name}  ({len(dataset)} samples) …")

    for idx in tqdm(range(len(dataset)), desc=dataset_name):
        imu_input, target_r6d, fname = dataset[idx]
        imu_input = imu_input.unsqueeze(0).to(device)
        seq_len = imu_input.shape[1]

        pred_r6d = model(imu_input, [seq_len])[0].cpu()      # (N, 144)
        target_r6d = target_r6d.cpu()

        # r6d → rotation matrix → axis-angle
        pred_rotmat = r6d_to_rotation_matrix(pred_r6d.view(-1, 24, 6))
        target_rotmat = r6d_to_rotation_matrix(target_r6d.view(-1, 24, 6))

        pred_aa = rotation_matrix_to_axis_angle(pred_rotmat.view(-1, 3, 3)).view(seq_len, 24, 3)
        target_aa = rotation_matrix_to_axis_angle(target_rotmat.view(-1, 3, 3)).view(seq_len, 24, 3)

        pred_orient = pred_aa[:, 0].numpy().astype(np.float32)
        pred_body   = pred_aa[:, 1:].numpy().astype(np.float32)
        gt_orient   = target_aa[:, 0].numpy().astype(np.float32)
        gt_body     = target_aa[:, 1:].numpy().astype(np.float32)

        zeros3  = np.zeros((seq_len, 3),  dtype=np.float32)
        zeros10 = np.zeros((seq_len, 10), dtype=np.float32)

        result = {
            "pred": {"orient": pred_orient, "transl": zeros3, "pose": pred_body, "betas": zeros10},
            "gt":   {"orient": gt_orient,   "transl": zeros3, "pose": gt_body,   "betas": zeros10},
        }

        save_file = output_path / f"{fname}.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(result, f)

    print(f"Saved {len(dataset)} predictions → {output_path}")


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate IMUPoser on real IMU test sets")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["dip", "imuposer_real", "dip_imuposer"],
                        help="Which test set(s) to evaluate on")
    parser.add_argument("--combo", type=str, default="global",
                        choices=list(amass_combos.keys()),
                        help="IMU combo configuration")
    parser.add_argument("--pretrained_ckpt", type=str, required=True,
                        help="Path to the pre-trained base IMUPoser checkpoint")
    parser.add_argument("--finetuned_ckpt", type=str, default=None,
                        help="Path to the fine-tuned checkpoint (omit with --no_finetune)")
    parser.add_argument("--no_finetune", action="store_true",
                        help="Evaluate the pre-trained model directly without fine-tuning")
    parser.add_argument("--output_dir", type=str, default="../../predictions",
                        help="Directory to save prediction pkl files")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── build config (mkdir=False so we don't create checkpoint dirs) ───────
    config = Config(
        model="GlobalModelIMUPoser",
        project_root_dir="../../",
        joints_set=amass_combos["global"],
        r6d=True,
        loss_type="mse",
        use_joint_loss=True,
        device="0",
        mkdir=False,
        dataset_name=args.dataset,
    )

    # ── load model ──────────────────────────────────────────────────────────
    if args.no_finetune:
        print(f"Loading pre-trained model: {args.pretrained_ckpt}")
        model = IMUPoserModel.load_from_checkpoint(args.pretrained_ckpt, config=config)
    else:
        assert args.finetuned_ckpt is not None, \
            "Provide --finetuned_ckpt or use --no_finetune"
        print(f"Loading pre-trained base: {args.pretrained_ckpt}")
        pretrained = IMUPoserModel.load_from_checkpoint(args.pretrained_ckpt, config=config)
        print(f"Loading fine-tuned model: {args.finetuned_ckpt}")
        config.model = "GlobalModelIMUPoserFineTuneRealIMU"
        model = IMUPoserModelFineTune.load_from_checkpoint(
            args.finetuned_ckpt, config=config, pretrained_model=pretrained,
        )

    model.to(device)
    model.eval()

    # ── build dataset(s) & run inference ────────────────────────────────────
    # If dataset_name is "dip_imuposer" we evaluate on each test set separately
    # so that metrics can be computed per-dataset.
    if args.dataset == "dip_imuposer":
        eval_sets = ["dip", "imuposer_real"]
    else:
        eval_sets = [args.dataset]

    for ds_name in eval_sets:
        eval_config = Config(
            model=config.model,
            project_root_dir="../../",
            joints_set=amass_combos["global"],
            r6d=True,
            loss_type="mse",
            use_joint_loss=True,
            device="0",
            mkdir=False,
            dataset_name=ds_name,
        )
        dataset = FullLengthRealIMUDataset(eval_config, combo_name=args.combo)
        save_predictions(model, dataset, device,
                         dataset_name=f"imuposer_finetune/{ds_name}_{args.combo}",
                         output_dir=args.output_dir)


if __name__ == "__main__":
    main()
