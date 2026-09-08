"""Train IMUPoser on IMU4D SMPL-X shards.

Example:
    python scripts/train.py --experiment imuposer_humoto --datasets humoto/v1 --max_epochs 50
    python scripts/train.py --experiment imuposer_ft_imuposer --datasets imuposer/v2 \
        --pretrained_ckpt checkpoints/<run>/<ckpt>.ckpt --lr 5e-5 --max_epochs 30
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imuposer.config import Config, amass_combos, parse_dataset_specs  # noqa: E402
from imuposer.datasets.utils import IMUPoserDataModule  # noqa: E402
from imuposer.models.utils import get_model  # noqa: E402
from imuposer.utils import get_parser  # noqa: E402


def main() -> None:
    """Parse arguments, build config / data / model, and run Lightning training."""
    args = get_parser().parse_args()
    seed_everything(42, workers=True)

    config = Config(
        experiment=f"{args.experiment}_{args.combo_id}",
        model="IMUPoser",
        project_root_dir=str(REPO_ROOT),
        joints_set=amass_combos[args.combo_id],
        loss_type="mse",
        r6d=True,
        device=args.device,
        use_joint_loss=not args.no_joint_loss,
        datasets=parse_dataset_specs(args.datasets),
        window_length=args.window_length,
        align_first_frame=not args.no_align,
    )
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.lr = args.lr
    print("Config:\n" + config.summary())

    model = get_model(config, pretrained_ckpt=args.pretrained_ckpt)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"IMUPoserModel: {num_params:,} parameters, input (B, T, {model.n_input}) -> output (B, T, {model.n_pose_output})")

    datamodule = IMUPoserDataModule(config)
    checkpoint_path = config.checkpoint_path
    if args.logger == "wandb":
        logger = WandbLogger(project=config.experiment, save_dir=str(checkpoint_path))
    else:
        logger = CSVLogger(save_dir=str(checkpoint_path), name="csv")

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-5, patience=5),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=10, dirpath=str(checkpoint_path),
                        save_weights_only=True, filename="epoch={epoch}-val_loss={val_loss:.5f}",
                        auto_insert_metric_name=False),
    ]
    accelerator, devices = ("cpu", 1) if config.device.type == "cpu" else ("gpu", [config.device.index])
    trainer = pl.Trainer(fast_dev_run=args.fast_dev_run, logger=logger, max_epochs=args.max_epochs,
                         accelerator=accelerator, devices=devices, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)

    if not args.fast_dev_run:
        with open(checkpoint_path / "best_model.txt", "w") as f:
            f.write(f"{callbacks[1].best_model_path}\n\n{callbacks[1].best_k_models}")
        print(f"best checkpoint: {callbacks[1].best_model_path}")


if __name__ == "__main__":
    main()
