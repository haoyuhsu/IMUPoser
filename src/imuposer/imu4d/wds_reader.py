"""Streaming reader for IMU4D WebDataset shards.

Each shard is a ``.tar`` whose members are pickled sample dicts (``<key>.sample.pkl``). The
reader streams tar members sequentially, so memory usage is bounded by the shuffle buffer and
no index is built. It is an :class:`~torch.utils.data.IterableDataset`; use it with a
``DataLoader(shuffle=False)`` and let the dataset do the shuffling.
"""

from __future__ import annotations

import json
import pickle
import random
import tarfile
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from imuposer.imu4d.paths import wds_root

SAMPLE_SUFFIX = ".sample.pkl"


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles written by numpy>=2 (``numpy._core``) under numpy 1.x (``numpy.core``)."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_manifest(wds_dir: Path) -> dict:
    """Read ``manifest.json`` of one WDS root."""
    with open(wds_dir / "manifest.json", "r") as f:
        return json.load(f)


def list_shards(wds_dir: Path, split: str) -> Tuple[List[Path], int]:
    """Return the sorted shard paths of ``split`` and its sample count (honours ``shared_with``)."""
    manifest = load_manifest(wds_dir)
    assert split in manifest["splits"], f"split {split!r} not in {list(manifest['splits'])}"
    info = manifest["splits"][split]
    shard_split = info.get("shared_with", split)
    shards = sorted((wds_dir / shard_split).glob(f"{shard_split}-*.tar"))
    assert len(shards) == info["num_shards"], f"{wds_dir}/{shard_split}: {len(shards)} != {info['num_shards']}"
    return shards, int(info["num_samples"])


def iter_shard(path: Path) -> Iterator[dict]:
    """Yield every pickled sample of one tar shard in file order (streaming, no seeks)."""
    with tarfile.open(path, mode="r|") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(SAMPLE_SUFFIX):
                continue
            sample = _NumpyCompatUnpickler(tf.extractfile(member)).load()
            sample["__key__"] = member.name[: -len(SAMPLE_SUFFIX)]
            yield sample


class IMU4DShardDataset(IterableDataset):
    """Stream samples from one or more IMU4D datasets and map them through ``transform``.

    ``dataset_specs`` are ``'<dataset>/<version>'`` strings (e.g. ``'humoto/v1'``); their shards are
    interleaved. ``transform(sample) -> item | None`` runs inside the worker; ``None`` drops the sample.
    Shard order is reshuffled every epoch with a seed shared by all workers, so the worker
    partition stays disjoint. No ``__len__`` is defined: each worker batches its own stream, so the
    batch count per epoch depends on the worker split; ``num_samples`` (from the manifests) is an upper
    bound on the items per epoch for progress reporting. When there are fewer shards than workers every worker streams all
    shards and keeps every ``num_workers``-th sample instead.
    """

    def __init__(
        self,
        dataset_specs: Sequence[str],
        split: str,
        transform: Callable[[dict], Optional[Any]],
        shuffle: bool = False,
        shuffle_buffer: int = 1000,
        seed: int = 0,
    ):
        super().__init__()
        assert len(dataset_specs) > 0, "need at least one dataset spec"
        self.dataset_specs = list(dataset_specs)
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self._epoch = 0

        self.entries: List[Tuple[Path, str, str]] = []  # (shard path, manifest source, spec)
        self.num_samples = 0
        for spec in self.dataset_specs:
            root = wds_root(spec)
            source = load_manifest(root)["source"]
            shards, count = list_shards(root, split)
            self.entries.extend((shard, source, spec) for shard in shards)
            self.num_samples += count

    def _raw_samples(self, shards: List[Tuple[Path, str, str]], stride: int, offset: int) -> Iterator[dict]:
        """Stream raw sample dicts from ``shards``, keeping indices congruent to ``offset`` mod ``stride``."""
        index = 0
        for path, source, spec in shards:
            for sample in iter_shard(path):
                if index % stride == offset:
                    sample["__source__"] = source
                    sample["__dataset__"] = spec
                    yield sample
                index += 1

    def __iter__(self) -> Iterator[Any]:
        """Yield transformed items for this worker's share of the current epoch."""
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        # torch.initial_seed() is base_seed + worker_id and base_seed changes per DataLoader epoch;
        # _epoch covers persistent workers, whose initial seed stays fixed.
        base_seed = torch.initial_seed() - worker_id
        shard_rng = random.Random(self.seed + base_seed + self._epoch)
        item_rng = random.Random(self.seed + base_seed + self._epoch + 7919 * (worker_id + 1))
        self._epoch += 1

        shards = list(self.entries)
        if self.shuffle:
            shard_rng.shuffle(shards)
        if len(shards) >= num_workers:
            my_shards, stride, offset = shards[worker_id::num_workers], 1, 0
        else:
            my_shards, stride, offset = shards, num_workers, worker_id

        items = (self.transform(s) for s in self._raw_samples(my_shards, stride, offset))
        items = (item for item in items if item is not None)
        if not self.shuffle:
            yield from items
            return

        buffer: List[Any] = []
        for item in items:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer:
                yield buffer.pop(item_rng.randrange(len(buffer)))
        item_rng.shuffle(buffer)
        yield from buffer
