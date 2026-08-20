"""YAML configuration loading into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    bayer_pattern: str = "RGGB"
    side: str = "left"


@dataclass
class PreprocessingConfig:
    crop_bottom_px: int = 175
    clahe_enabled: bool = False
    clahe_clip_limit: float = 10.0
    clahe_tile_grid_size: int = 8
    gaussian_blur_enabled: bool = False
    gaussian_blur_ksize: int = 5
    gaussian_blur_sigma: float = 0.0


@dataclass
class TrackingConfig:
    max_corners: int = 1000
    grid_rows: int = 2
    grid_cols: int = 2
    hysteresis_margin: float = 0.15
    hysteresis_radius: float = 5.0


@dataclass
class DiskConfig:
    device: str = "cuda"
    checkpoint: str = "depth"
    window_size: int = 5
    score_threshold: float = 0.0
    match_max_distance: float = 0.5
    kp_oversample_factor: int = 3


@dataclass
class LightGlueConfig:
    device: str = "cuda"
    feature_name: str = "disk"
    min_confidence: float = 0.9


@dataclass
class StereoConfig:
    min_disparity_px: float = 1.0
    max_depth_m: float = 60.0
    pixel_sigma: float = 1.0    # rectified-pixel reprojection noise for GenericStereoFactor3D
    huber_k: float = 1.345      # standard Huber constant (~95% efficiency under Gaussian noise)
    landmark_prior_sigma: float = 3.0  # m, weak prior anchoring each new landmark near its initial triangulation


@dataclass
class MotionPriorConfig:
    rotation_sigma: float = 0.05
    translation_sigma: float = 0.05
    angular_velocity_rw_sigma: float = 2.0
    linear_velocity_rw_sigma: float = 5.0
    initial_velocity_prior_sigma: float = 1.0
    initial_pose_prior_sigma: float = 1.0e-3


@dataclass
class GraphConfig:
    vo_lookback: int = 4
    # gtsam_unstable.IncrementalFixedLagSmoother's window, in seconds: any
    # pose/velocity/landmark variable not re-touched within this many seconds
    # of the newest timestamp gets properly marginalized (not just dropped).
    # Must comfortably exceed vo_lookback frames' worth of real elapsed time
    # for the slowest-fps dataset in use, or active variables could be
    # marginalized out from under a still-in-window frame.
    smoother_lag_s: float = 1.0


@dataclass
class VisualizationConfig:
    enabled: bool = False
    output_path: str | None = None  # set by run_trajectory.py; None disables even if enabled=True
    fps: int = 8
    low_info_threshold: int = 20  # n_landmark_observations below this -> frame flagged LOW-INFO


_VALID_MODES = {"stereo", "mono"}


@dataclass
class Config:
    mode: str
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    tracking: TrackingConfig
    disk: DiskConfig
    lightglue: LightGlueConfig
    stereo: StereoConfig
    motion_prior: MotionPriorConfig
    graph: GraphConfig
    visualization: VisualizationConfig

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        mode = raw.get("mode", "stereo")
        if mode not in _VALID_MODES:
            raise ValueError(f"config `mode` must be one of {_VALID_MODES}, got {mode!r}")
        return Config(
            mode=mode,
            dataset=DatasetConfig(**raw.get("dataset", {})),
            preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
            disk=DiskConfig(**raw.get("disk", {})),
            lightglue=LightGlueConfig(**raw.get("lightglue", {})),
            stereo=StereoConfig(**raw.get("stereo", {})),
            motion_prior=MotionPriorConfig(**raw.get("motion_prior", {})),
            graph=GraphConfig(**raw.get("graph", {})),
            visualization=VisualizationConfig(**raw.get("visualization", {})),
        )
