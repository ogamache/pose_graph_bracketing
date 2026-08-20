"""Loading of ROS-style stereo camera calibration YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class CameraCalibration:
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (n,)
    image_width: int
    image_height: int


def load_stereo_calibration(path: str | Path) -> CameraCalibration:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    K = np.array(raw["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    dist = np.array(raw["distortion_coefficients"]["data"], dtype=np.float64)

    return CameraCalibration(
        K=K,
        dist=dist,
        image_width=int(raw["image_width"]),
        image_height=int(raw["image_height"]),
    )


def crop_bottom_calibration(calib: CameraCalibration, crop_bottom_px: int) -> CameraCalibration:
    """Return a copy of the calibration adjusted for a bottom-rows crop.

    Cropping rows off the bottom does not change the principal point or focal
    length; only the effective image height shrinks.
    """
    if crop_bottom_px <= 0:
        return calib
    return CameraCalibration(
        K=calib.K.copy(),
        dist=calib.dist.copy(),
        image_width=calib.image_width,
        image_height=calib.image_height - crop_bottom_px,
    )
