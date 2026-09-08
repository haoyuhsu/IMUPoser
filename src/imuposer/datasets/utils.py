"""Dataset construction, padding collate, and the Lightning data module."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from imuposer.config import Config
from imuposer.datasets.imu4dDataset import IMU4DDataset


def get_dataset(config: Config, split: str, combo: Optional[str] = None, return_meta: bool = False) -> IMU4DDataset:
    """Build the streaming dataset for one IMU4D split (``train`` / ``val`` / ``test``)."""
    return IMU4DDataset(split, config, combo=combo, return_meta=return_meta)


def pad_seq(batch: List[tuple]):
    """Pad variable-length clips; returns ``(inputs (B,T,60), outputs (B,T,C), input_lens, output_lens)``."""
    inputs = [item[0] for item in batch]
    outputs = [item[1] for item in batch]
    input_lens = [x.shape[0] for x in inputs]
    output_lens = [y.shape[0] for y in outputs]
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    outputs = nn.utils.rnn.pad_sequence(outputs, batch_first=True)
    return inputs, outputs, input_lens, output_lens


class IMUPoserDataModule(pl.LightningDataModule):
    """Train / val / test loaders over the streaming IMU4D datasets (shuffling happens in the dataset)."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate the split datasets; shards are only opened when iterated."""
        self.train_dataset = get_dataset(self.config, "train")
        self.val_dataset = get_dataset(self.config, "val")
        self.test_dataset = get_dataset(self.config, "test")
        print(f"IMU4D datasets {self.config.datasets}: train={self.train_dataset.num_samples} "
              f"val={self.val_dataset.num_samples} test={self.test_dataset.num_samples} samples")

    def _loader(self, dataset: IMU4DDataset) -> DataLoader:
        """DataLoader for an IterableDataset: no sampler-side shuffle, persistent workers."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=pad_seq,
            num_workers=self.config.num_workers,
            shuffle=False,
            persistent_workers=self.config.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset)
