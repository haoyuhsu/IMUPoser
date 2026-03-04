"""
Fine-tune a pre-trained IMUPoser model on real IMU datasets (DIP-IMU and/or IMUPoser).

Usage examples:

  # Fine-tune on DIP-IMU only
  python "2. FineTune RealIMU.py" \
      --combo_id global \
      --experiment IMUPoserFineTuneRealIMU \
      --pretrained_ckpt /path/to/pretrained.ckpt \
      --dataset_name dip \
      --max_epochs 30

  # Fine-tune on IMUPoser real-world only
  python "2. FineTune RealIMU.py" \
      --combo_id global \
      --experiment IMUPoserFineTuneRealIMU \
      --pretrained_ckpt /path/to/pretrained.ckpt \
      --dataset_name imuposer_real \
      --max_epochs 30

  # Fine-tune on both DIP-IMU + IMUPoser combined
  python "2. FineTune RealIMU.py" \
      --combo_id global \
      --experiment IMUPoserFineTuneRealIMU \
      --pretrained_ckpt /path/to/pretrained.ckpt \
      --dataset_name dip_imuposer \
      --max_epochs 30
"""

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser

seed_everything(42, workers=True)

# ── parse args ──────────────────────────────────────────────────────────────
parser = get_parser()
args = parser.parse_args()

combo_id = args.combo_id
fast_dev_run = args.fast_dev_run
_experiment = args.experiment
train_epochs = int(args.max_epochs)

assert args.pretrained_ckpt is not None, \
    "Please provide --pretrained_ckpt pointing to a pre-trained IMUPoser checkpoint."
assert args.dataset_name in ("dip", "imuposer_real", "dip_imuposer"), \
    f"--dataset_name must be one of: dip, imuposer_real, dip_imuposer  (got '{args.dataset_name}')"

# ── config ──────────────────────────────────────────────────────────────────
config = Config(
    experiment=f"{_experiment}_{combo_id}",
    model="GlobalModelIMUPoser",          # start with base model for loading ckpt
    project_root_dir="../../",
    joints_set=amass_combos[combo_id],
    normalize="no_translation",
    r6d=True,
    loss_type="mse",
    use_joint_loss=True,
    device="0",
    dataset_name=args.dataset_name,
)

# ── load pre-trained model ──────────────────────────────────────────────────
print(f"Loading pre-trained checkpoint: {args.pretrained_ckpt}")
pretrained_model = IMUPoserModel.load_from_checkpoint(
    args.pretrained_ckpt,
    config=config,
)

# ── switch to fine-tune model & dataset ─────────────────────────────────────
config.model = "GlobalModelIMUPoserFineTuneRealIMU"
model = get_model(config, pretrained=pretrained_model)
datamodule = get_datamodule(config)
checkpoint_path = config.checkpoint_path

# ── trainer ─────────────────────────────────────────────────────────────────
wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

early_stopping_callback = EarlyStopping(
    monitor="validation_step_loss", mode="min", verbose=False,
    min_delta=0.00001, patience=5,
)
checkpoint_callback = ModelCheckpoint(
    monitor="validation_step_loss", mode="min", verbose=False,
    save_top_k=10, dirpath=checkpoint_path, save_weights_only=True,
    filename="epoch={epoch}-val_loss={validation_step_loss:.5f}",
)

trainer = pl.Trainer(
    fast_dev_run=fast_dev_run,
    logger=wandb_logger,
    max_epochs=train_epochs,
    accelerator="gpu",
    devices=[0],
    callbacks=[early_stopping_callback, checkpoint_callback],
    deterministic=True,
)

# ── train ───────────────────────────────────────────────────────────────────
trainer.fit(model, datamodule=datamodule)

# ── save best checkpoint path ──────────────────────────────────────────────
with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")

print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
