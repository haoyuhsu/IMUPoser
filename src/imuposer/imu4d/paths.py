"""Resolve IMU4D repository, data, and body-model locations for this baseline.

Nothing here writes outside the baseline directory. Every path can be overridden by an
environment variable so the baseline also works when checked out elsewhere:

    IMU4D_ROOT        IMU4D repository (provides ``imu_synthesis``); default: the parent checkout
    IMU4D_DATA_ROOT   data tree with ``processed/<dataset>/<version>/wds``; default: ``$IMU4D_ROOT/data``
    SMPLX_MODEL_PATH  ``SMPLX_NEUTRAL.npz``; default: first existing candidate listed in
                      :func:`smplx_model_path`
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# baseline/IMUPoser/src/imuposer/imu4d/paths.py -> IMU4D_dev
_DEFAULT_IMU4D_ROOT = Path(__file__).resolve().parents[5]


def imu4d_root() -> Path:
    """Return the IMU4D repository root (must contain ``imu_synthesis/``)."""
    root = Path(os.environ.get("IMU4D_ROOT", _DEFAULT_IMU4D_ROOT)).expanduser().resolve()
    assert (root / "imu_synthesis").is_dir(), f"{root} has no imu_synthesis/; set IMU4D_ROOT"
    return root


def imu4d_data_root() -> Path:
    """Return the IMU4D data root (contains ``processed/`` and optionally ``models/``)."""
    return Path(os.environ.get("IMU4D_DATA_ROOT", imu4d_root() / "data")).expanduser().resolve()


def wds_root(dataset_spec: str) -> Path:
    """Map ``'humoto/v1'`` to ``<data_root>/processed/humoto/v1/wds`` and check the manifest exists."""
    root = imu4d_data_root() / "processed" / dataset_spec / "wds"
    assert (root / "manifest.json").is_file(), f"no manifest.json under {root}"
    return root


def smplx_model_path() -> Path:
    """Return the SMPL-X neutral ``.npz``; the first existing candidate wins."""
    candidates = []
    for key in ("SMPLX_MODEL_PATH", "IMU4D_SMPLX_PATH"):
        if os.environ.get(key):
            candidates.append(Path(os.environ[key]).expanduser())
    candidates.append(imu4d_data_root() / "models" / "smplx" / "SMPLX_NEUTRAL.npz")
    candidates.append(
        imu4d_root().parent / "body_models" / "human_model_files" / "smplx" / "SMPLX_NEUTRAL.npz"
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "SMPLX_NEUTRAL.npz not found; set SMPLX_MODEL_PATH. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def ensure_imu_synthesis_importable() -> None:
    """Append the IMU4D root to ``sys.path`` so ``imu_synthesis`` imports as a package.

    Appending (not inserting first) keeps the baseline's own top-level packages (``utils``, ``data``)
    ahead of the identically named directories in the IMU4D checkout."""
    root = str(imu4d_root())
    if root not in sys.path:
        sys.path.append(root)
