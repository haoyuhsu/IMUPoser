"""Model construction helpers."""

from __future__ import annotations

from typing import Optional

from imuposer.config import Config
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel


def get_model(config: Config, pretrained_ckpt: Optional[str] = None) -> IMUPoserModel:
    """Build ``IMUPoserModel``; when ``pretrained_ckpt`` is given, load its weights and fine-tune from them."""
    if pretrained_ckpt is None:
        return IMUPoserModel(config=config)
    model = IMUPoserModel.load_from_checkpoint(pretrained_ckpt, config=config)
    model.lr = config.lr
    return model
